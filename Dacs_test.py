import torch

# primary_model = torch.nn.Linear(1, 1)
# teacher_model = torch.nn.Linear(1, 1)
#
# primary_model.weight.data = torch.tensor([2.0])
#
# def init_ema_weights(teacher_model, primary_model):
#     for param in teacher_model.parameters():
#         param.detach_()
#
#     primary_params = list(primary_model.parameters())
#     teacher_params = list(teacher_model.parameters())
#
#     for i in range(len(primary_params)):
#         print(teacher_params[i].data.shape)
#
#         if not teacher_params[i].data.shape:
#             teacher_params[i].data = primary_params[i].data.clone()
#         else:
#             teacher_params[i].data[:] = primary_params[i].data[:].clone()
#
# init_ema_weights(teacher_model, primary_model)
#
# print("Teacher Model's Weight:", teacher_model.weight.data)
#
# # Initialize the teacher value
# teacher_value = 10.0
#
# # Initialize the primary model's value
# primary_value = 5.0
#
# # Initialize alpha_teacher
# alpha_teacher = 0.1
#
# # Update the teacher value using EMA
# ema_value = alpha_teacher * teacher_value + (1 - alpha_teacher) * primary_value
#
# # Print the updated value
# print("Updated EMA Value:", ema_value)


# Sample log_vars dictionary
log_vars = {
    'loss': 0.5,
    'accuracy': 0.85,
    'learning_rate': 0.001
}

# Check if 'loss' is unnecessary
# if 'loss' in log_vars:
#     # Perform some operations with 'loss'
#     loss_value = log_vars['loss']
#     print("Loss:", loss_value)
#
# # Remove 'loss' from log_vars (because it's not needed later)
# log_vars.pop('loss', None)
#
# # Now log_vars no longer contains 'loss'
# print("Updated log_vars:", log_vars)
#
# outputs = dict(
#             log_vars=log_vars, num_sample=len([1,2,3])
#         )
# print(outputs)


# import torch
#
# # Create two feature maps
# f1 = torch.tensor([[1.0, 2.0, 3.0],
#                    [4.0, 5.0, 6.0]])
# f2 = torch.tensor([[2.0, 3.0, 4.0],
#                    [5.0, 6.0, 7.0]])
#
# # Calculate the feature difference
# feat_diff = f1 - f2
#
# # Calculate the pairwise feature distances
# pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
#
# # Create a mask to consider only the second row of features
# mask = torch.tensor([0, 1], dtype=torch.bool)
#
# # Apply the mask to the pairwise feature distances
# pw_feat_dist_masked = pw_feat_dist[mask]
#
# # Calculate the mean of the masked feature distances
# mean_feat_dist = torch.mean(pw_feat_dist_masked)
#
# print("Feature Difference:")
# print(feat_diff)
# print("Pairwise Feature Distances:")
# print(pw_feat_dist)
# print("Masked Pairwise Feature Distances:")
# print(pw_feat_dist_masked)
# print("Mean Feature Distance (masked):", mean_feat_dist.item())

import torch

# Create a sample tensor
tensor = torch.tensor([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ]
])

# Access the last dimension using -1
last_dim_size = tensor.shape[-1]

print("Tensor Shape:", tensor.shape)
print("Size of Last Dimension (using -1):", last_dim_size)

