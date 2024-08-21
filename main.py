from discriminator import Discriminator
from generator import Generator
import torch
import torch.nn as nn
import torch.optim as optim


generator = Generator((256, 256), (128, 128))
discriminator = Discriminator((256, 256), (128, 128))
optim1 = optim.SGD(generator.parameters(), lr=0.005)
optim2 = optim.SGD(discriminator.parameters(), lr=0.005)
# training loop
epochs = 10
generatorCriterion = nn.MSELoss()
discriminatorCriterion = nn.BCELoss()
for i in range(epochs):
    for i in range(len(lowResTrainPixels)):
        realDataHigh = highResTrainPixels[i].unsqueeze(0)  # TODO with real data
        realDataLow = lowResTrainPixels[i].unsqueeze(0)
        fakeData = generator(realDataLow)
        realLabels = torch.ones(1, 1)
        fakeLabels = torch.zeros(1, 1)
        labels = torch.cat((realLabels, fakeLabels), dim=0)
        data = torch.cat((realDataHigh, fakeData), dim=0)

        # discriminator train
        optim2.zero_grad()
        output_d = discriminator(data)
        loss_d = discriminatorCriterion(output_d, labels)
        loss_d.backward()
        optim2.step()

        # generator step
        discriminatorPredictions = discriminator(fakeData)

        optim1.zero_grad()
        loss_generator = generatorCriterion(
            fakeData, realDataHigh
        ) + discriminatorCriterion(
            discriminatorPredictions, torch.ones_like(discriminatorPredictions)
        )
        loss_generator.backward()
        optim2.step()
    print(f"Epoch {i+1}/{epochs} completed.")
