import torch
import torch.nn as nn
import tqdm
import numpy as np


class VRPPointerNetwork(nn.Module):
    def __init__(self, vehicle_capacity: int):
        super(VRPPointerNetwork, self).__init__()
        self.vehicle_capacity = vehicle_capacity
        self.actor = Actor(vehicle_capacity=self.vehicle_capacity)

    def forward(self, x: tuple[tuple, float, float]):
        # location, demand, depot

        decoder_points, all_log_probabilities, static_location_embedding = self.actor(x)

        return decoder_points, all_log_probabilities, static_location_embedding


# an actor network that predicts a probability distribution over the next action at any given decision step,
class Actor(nn.Module):
    # Consists of encoder and decoder
    def __init__(self, vehicle_capacity: int):
        super(Actor, self).__init__()
        self.vehicle_capacity = vehicle_capacity
        self.encoder: Encoder = Encoder()
        self.decoder: DecoderRNN = DecoderRNN()

    def forward(self, x: tuple[tuple, float, float]):
        # x - (location,demand,depot)
        (encoded_output, depot_embedding) = self.encoder(x, self.vehicle_capacity)
        decoder_points, all_log_probabilities = self.decoder(
            input_data=x,
            encoder_outputs=(encoded_output, depot_embedding),
            vehicle_capacity=self.vehicle_capacity,
        )
        return decoder_points, all_log_probabilities, encoded_output[0]


# no recalculation in encoder
class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        dropout: float = 0.0,
    ):
        super(Encoder, self).__init__()
        self.static = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=1
        )
        self.dynamic = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=1
        )

    def forward(self, input: tuple[tuple, float], vehicle_capacity: float):
        # input - (location,demand,depot)
        location, demand, depot = input

        static_encoded = self.static(location.permute(0, 2, 1))  # expanding 2 points
        # number of dimensions is number of channels is Conv
        depot_embedding = self.static(
            depot.unsqueeze(1).permute(0, 2, 1)
        )  # required by decoder for input token
        dynamic_encoded = self.dynamic(
            torch.stack((demand, vehicle_capacity - demand), axis=1)
        )  # dynamic so that we can apply mask
        # Combining both branch - not present in the paper
        encoded_output = (static_encoded, dynamic_encoded)
        return (
            encoded_output,
            depot_embedding,
        )  # not applying conv1d as paper suggested


class Attention(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int = 2 * 128,
        decoder_hidden_size: int = 128,
    ):
        super(Attention, self).__init__()
        # self.vehicle_capacity = vehicle_capacity
        self.tanh = torch.tanh
        self.softmax = torch.softmax
        # 32 is dimensionality reduction. Paper does not talk apart from V being learnable and becomes a vector
        self.W1_encoder = nn.Linear(encoder_hidden_size, 128)
        self.W2_decoder = nn.Linear(decoder_hidden_size, 128)
        self.W_wt = nn.Linear(decoder_hidden_size, 128)
        self.W_ct = nn.Linear(decoder_hidden_size, 128)
        self.Wa = nn.Linear(decoder_hidden_size, 32)
        self.Va = nn.Linear(32, 1)  # V is a vector

        self.Wc = nn.Linear(decoder_hidden_size, 32)
        self.Vc = nn.Linear(decoder_hidden_size, 1)  # V is a vector

    def forward(
        self,
        decoder_output: torch.tensor,
        encoder_outputs: torch.tensor,
        remaining_truck_load: list[float],
        remaining_customer_demand: list[float],
        vehicle_capacity,
    ):
        # Calculating attention from the lecture video; paper is not clear
        # appending encoder with decoder will increase the size
        x_t = self.W1_encoder(
            torch.cat(encoder_outputs, axis=1).permute(0, 2, 1)
        ) + self.W2_decoder(
            decoder_output
        )  # [10, 6, 128]

        tanh_ = self.tanh(self.Wa(x_t))
        u_t = self.Va(tanh_).squeeze(-1)  # torch.nn.Linear(in_features, out_features)
        a_t = self.softmax(u_t, axis=1)  # [10, 6] attention for each row
        c_t = torch.sum(a_t.unsqueeze(-1) * x_t, axis=1)

        w_c = self.W_wt(x_t) + self.W_ct(c_t).unsqueeze(1)

        logits = self.Vc(self.tanh(w_c)).squeeze(-1)
        # [10, 6]

        softmax_ = self.softmax(logits, axis=1)  # CCE requires unnormalized logits
        log_probabilities = torch.log(softmax_)

        # (i) nodes with zero demand are not allowed to be visited;
        # # so that even with remaining capacity of truck, it will not move anywhere else
        # if torch.any(remaining_customer_demand < 0):
        #     print("Here")
        self.zero_demand_mask = (
            remaining_customer_demand[:, 1:] == 0
        )  # depot is never masked
        log_probabilities[:, 1:] = log_probabilities[:, 1:].masked_fill(
            self.zero_demand_mask, float("-inf")
        )
        remaining_truck_load = remaining_truck_load.to("mps")
        # (ii) all customer nodes will be masked if the vehicle’s remaining load is exactly 0;
        self.remaining_load_mask = (remaining_truck_load == 0).unsqueeze(
            1
        )  # if a truck has no more load, mask all customer cities, so that it returns to depot
        # reset vehicle capacity is remaining load is 0 and currently in depot
        log_probabilities[:, 1:] = log_probabilities[:, 1:].masked_fill(
            self.remaining_load_mask, float("-inf")
        )
        # (iii) the customers whose demands are greater than the current vehicle load are masked.
        # log of infinity is infinity
        # if all_masked_fill:
        #     go back to depot
        #     capacity is full . it will impact input state of next decoding step

        self.demand_exceeds_load_mask = (
            remaining_customer_demand[:, 1:] > remaining_truck_load.unsqueeze(1)
        ).unsqueeze(1)
        log_probabilities[:, 1:] = log_probabilities[:, 1:].masked_fill(
            self.remaining_load_mask, float("-inf")
        )

        maxvalue, pointer = log_probabilities.max(axis=1)
        # Update Masks

        for i in range(len(pointer)):
            # Return to depot flag - if 0, then distance calculation will change
            if pointer[i] == 0:
                remaining_truck_load[i] = vehicle_capacity
                continue
            # Truck Load
            remaining_truck_load[i] = (
                remaining_truck_load[i] - remaining_customer_demand[i][pointer[i]]
            )
            # Customer remaining demand #Only allowing full demand
            remaining_customer_demand[i][pointer[i]] = (
                remaining_customer_demand[i][pointer[i]] - remaining_truck_load[i]
            )

        return (
            log_probabilities,
            pointer,
            remaining_truck_load,
            remaining_customer_demand,
        )


class DecoderRNN(nn.Module):
    # Use only static
    def __init__(
        self, input_size: int = 128, hidden_size: int = 128, dropout: int = 0.1
    ):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.LSTMCell = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        # random uniform weight initialization from -0.08 to 0.08
        for name, param in self.LSTMCell.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, a=-0.08, b=0.08)

    def forward(
        self,
        input_data,
        encoder_outputs,
        vehicle_capacity: float,
    ):
        location, demand, depot_embedding = input_data
        ((static_embedding, dynamic_embedding), depot_embedding) = encoder_outputs
        batch_size, input_dimenions, n_steps = static_embedding.shape

        # Input the start token
        # We assume that the vehicle is located at the depot at time 0,
        # so the first input to the decoder is an embedding of the depot location.
        depot_embedding = depot_embedding.permute(0, 2, 1)
        decoder_input = depot_embedding
        decoder_outputs = []

        remaining_truck_load = torch.tensor([vehicle_capacity] * batch_size).to("mps")

        remaining_customer_demand = demand
        h0 = (
            torch.zeros(1, decoder_input.size(0), self.hidden_size)
            .requires_grad_()
            .to("mps")
        )

        # Initialize cell state
        c0 = (
            torch.zeros(1, decoder_input.size(0), self.hidden_size)
            .requires_grad_()
            .to("mps")
        )

        decoder_hidden = (h0, c0)
        all_log_probabilities = []
        for counter in range(100):
            # Decode next

            (decoder_output, decoder_hidden) = self.LSTMCell(
                decoder_input,
                decoder_hidden,  # required for batched as per pytorch documentation
            )

            attention = Attention().to("mps")
            (
                log_probabilities,
                batch_pointers,
                remaining_truck_load,
                remaining_customer_demand,
            ) = attention(
                decoder_output=decoder_output,
                encoder_outputs=(static_embedding, dynamic_embedding),
                remaining_truck_load=remaining_truck_load,
                remaining_customer_demand=remaining_customer_demand,
                vehicle_capacity=vehicle_capacity,
            )
            decoder_outputs.append(batch_pointers)
            next_input = torch.zeros((batch_size, static_embedding.shape[1])).to("mps")
            next_input = next_input.unsqueeze(
                1
            )  # torch.Size([4, 1, 2]) placeholder for storing next input
            # row_ = []
            for i in range(len(next_input)):
                next_input[i, :] = static_embedding.permute(0, 2, 1)[i][
                    batch_pointers[i]
                ]
                # row_.append(log_probabilities[i][batch_pointers[i]])
            all_log_probabilities.append(log_probabilities)
            decoder_input = next_input

            ## categorical cross entropy - how to use labels?
            # Use pointer as input for next roll
            # https://stackoverflow.com/questions/61187520/should-decoder-prediction-be-detached-in-pytorch-training
            # decoder_input = topi.squeeze(
            #     -1
            # ).detach()  # detach from history as input for next roll

        decoder_points = torch.stack(
            decoder_outputs, axis=1
        )  # 100 cities for each in batch.

        all_log_probabilities = torch.stack(all_log_probabilities, axis=1)
        return decoder_points, all_log_probabilities


# a critic network that estimates the reward for any problem instance from a given state
class Critic(nn.Module):
    # output probabilities of actor to compute weighted sum of embedded inputs
    def __init__(self, hidden_size=10):
        super(Critic, self).__init__()
        # numbe of iterations
        self.dense = nn.Linear(128, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.relu = torch.relu

    def forward(self, log_probabilities, location_embedding):
        out = None
        # [3, 100, 6]
        # [3, 6, 128] embedding for each batch
        # https://discuss.pytorch.org/t/weighted-sum-of-matrices/97961
        location_embedding = location_embedding.permute(0, 2, 1)
        # [3,100,128]
        # use the output probabilities of the actor network (not log probability-converts mask to 0)
        # to compute a weighted sum of the embedded inputs

        weighted_sum = torch.bmm(
            torch.softmax(log_probabilities, axis=2), location_embedding
        )

        # Applying non linearity
        out = self.dense(weighted_sum)
        # [3, 100, 32]
        out = self.relu(out)

        # Converting to single value
        out = self.linear(out)
        # [3,100]

        # We also update the critic network in step 15 in the direction of reducing the difference between the expected rewards with the observed ones
        # during Monte Carlo roll-outs
        # so for 100 steps, we take difference from actual 100 edge distances
        return out.squeeze(-1)
