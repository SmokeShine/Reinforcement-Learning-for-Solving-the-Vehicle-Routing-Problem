import os
import time
import logging
import logging.config
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def calculatedistance(
    decoded_points: list[list], alldistanceMatrix: list[np.array]
) -> list:
    rewards = torch.zeros((len(decoded_points), len(decoded_points[0]))).to("mps")
    completebatch = []
    for i, decoded_city in enumerate(decoded_points):
        # 0 to 0 will not matter as matrix is 0
        # depot is part of selection cities
        distanceMatrix = alldistanceMatrix[i]
        zero_decoded_city = torch.zeros((len(decoded_city) + 1)).long()

        zero_decoded_city[1:] = decoded_city
        row_edge_distances = distanceMatrix[
            zero_decoded_city[:-1], zero_decoded_city[1:]
        ]
        completebatch.append(row_edge_distances)
    batch_rewards = torch.stack(completebatch, axis=1)
    # reward is 0 at the start
    rewards = batch_rewards.permute(1, 0)
    return rewards


# Boiler Plate Code From BD4H and DL Class for recording metrics
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(
    actor,
    critic,
    device,
    data_loader,
    criterion,
    optimizer_actor,
    optimizer_critic,
    epoch,
    print_freq=50,
):
    actor.train()
    critic.train()
    losses_actor = AverageMeter()
    losses_critic = AverageMeter()
    total_losses = AverageMeter()
    for i, (location, demand, depot, distanceMatrix) in enumerate(data_loader):
        location = location.to(device)
        demand = demand.to(device)
        depot = depot.to(device)
        distanceMatrix = distanceMatrix.to(device)
        # distanceMatrix = torch.from_numpy(distanceMatrix.as_type(np.float32)).to(device)  # Space issue

        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        output_actor_points, output_actor_prob, static_location_embedding = actor(
            (location, demand, depot)
        )
        output_critic = critic(
            location_embedding=static_location_embedding,
            log_probabilities=output_actor_prob,
        )
        # target
        rewards = calculatedistance(
            decoded_points=output_actor_points, alldistanceMatrix=distanceMatrix
        )
        advantage = rewards - output_critic
        pointer_reward = torch.max(output_actor_prob, axis=2)
        # actor loss
        # https://stackoverflow.com/questions/65815598/calling-backward-function-for-two-different-neural-networks-but-getting-retai
        loss_actor = torch.mean(torch.sum(advantage * pointer_reward[0], axis=1))

        # critic loss
        loss_critic = torch.mean(torch.sum(torch.square(advantage), axis=1))
        assert not np.isnan(loss_actor.item()), "Actor diverged with loss = NaN"
        assert not np.isnan(loss_critic.item()), "Critic diverged with loss = NaN"

        # Calculating total loss
        total_loss = loss_actor + loss_actor
        # calculating loss for end leaves of graph for both actor and critic
        total_loss.backward()
        # Clipping
        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=2.0, norm_type=2)
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=2.0, norm_type=2)
        # propogate loss back to critic graph
        optimizer_critic.step()
        # propogate loss back to actor graph
        optimizer_actor.step()
        # Updating Averagemeter
        losses_actor.update(loss_actor.item(), output_critic.size(0))
        losses_critic.update(loss_critic.item(), output_critic.size(0))
        if i % print_freq == 0:
            logger.info(
                f"Epoch: {epoch} \t iteration: {i} \t Training Loss Actor Current:{losses_actor.val:.4f} Average:({losses_actor.avg:.4f})"
            )
            logger.info(
                f"Epoch: {epoch} \t iteration: {i} \t Training Loss Critic Current:{losses_critic.val:.4f} Average:({losses_critic.avg:.4f})"
            )

    return [losses_actor.avg, losses_critic.avg]


def evaluate(model, device, data_loader, criterion, optimizer, print_freq=10):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), target.size(0))
            if i % print_freq == 0:
                logger.info(
                    f"Validation Loss Current:{losses.val:.4f} Average:({losses.avg:.4f})"
                )
    return losses.avg


def save_checkpoint(model, optimizer, path):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, path)
    torch.save(model, "./checkpoint_model.pth", _use_new_zipfile_serialization=False)
    logger.info(f"checkpoint saved at {path}")
