import matplotlib.pyplot as plt

# Plot some sample images from the dataset

sample_loader = DataLoader(train_dataset,
batch_size=5, shuffle=True)

data_iter = iter(sample_loader)

images, labels = next(data_iter)

# Make predictions using the trained model

model.eval()

with torch.no_grad():

  images = images.view(-1, input_size)
  predictions = model(images)
  _, predicted_labels = torch.max(predictions, 1)

  # Plot the sample images with their predicted labels

plt.figure(figsize=(12, 6))

for i in range(5):

  plt.subplot(1, 5, i + 1)
  plt.imshow(images[i].view(28, 28).cpu().numpy(), cmap='gray')
  plt.title(f"Label: {labels[i]}, Predicted: {predicted_labels[i]}")
  plt.axis('off')
  plt.show()

# Save the trained model

torch.save(model.state_dict(), 'simple_nn_model.pth')
