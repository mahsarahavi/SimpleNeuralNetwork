import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


file_path = "datasset.csv"
df = pd.read_csv(file_path)
df_numeric = df.drop(columns=['name', 'Target'])
df_target = df['Target']
X_train, X_test, y_train, y_test = train_test_split(df_numeric, df_target, test_size = 0.2, random_state = 42)

X_train_tensor = torch.tensor(X_train.values, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype = torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype = torch.float32).view(-1, 1)

loss_functions = {
    "MSELoss": nn.MSELoss(),
    "L1Loss": nn.L1Loss(),
    "HuberLoss": nn.HuberLoss(),
    "SmoothL1Loss": nn.SmoothL1Loss()
}
optimizers = {
    "SGD": lambda params: optim.SGD(params, lr=0.01),
    "SGD + Momentum": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
    "Adam": lambda params: optim.Adam(params, lr=0.01),
    "RMSprop": lambda params: optim.RMSprop(params, lr=0.01),
    "AdamW": lambda params: optim.AdamW(params, lr=0.01)
}

def train_eval(loss_name, loss_fn, optimizer_name, optimizer_fn):
    input_size = X_train_tensor.shape[1]
    model = nn.Sequential(
        nn.Linear(input_size, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    optimizer = optimizer_fn(model.parameters())
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train_tensor)
        loss = loss_fn(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%100 == 0:
            print(f"epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_tensor)
        y_test_pred = model(X_test_tensor)
        test_loss = loss_fn(y_test_pred, y_test_tensor).item()
        
    print(f"{loss_name} Loss with {optimizer_name} Optimizer: {test_loss:.4f}\n")
    return test_loss, y_train_pred, y_test_pred


results = {}
for loss_name, loss_fn in loss_functions.items():
    for optimizer_name, optimizer_fn in optimizers.items():
        print(f"Training with {loss_name} loss and {optimizer_name} optimizer...")
        test_loss, y_train_pred, y_test_pred = train_eval(loss_name, loss_fn, optimizer_name, optimizer_fn)
        results[(loss_name, optimizer_name)] = test_loss


best_result = min(results)
best_loss_fn = best_result[0]
best_optimizer_fn = best_result[1]
best_loss = results[best_result]
print(f"Best Loss Function: {best_loss_fn}")
print(f"Best Optimizer: {best_optimizer_fn}")
print(f"Minimum Loss: {best_loss}")


best_loss, best_y_train, best_y_test = train_eval(best_loss_fn, loss_functions[best_loss_fn], best_optimizer_fn, optimizers[best_optimizer_fn])


df_train = X_train.copy()
df_train["Actual_Target"] = y_train.values
df_train["Predicted_Target"] = best_y_train
df_test = X_test.copy()
df_test["Actual_Target"] = y_test.values
df_test["Predicted_Target"] = best_y_test
df_with_predictions = pd.concat([df_train, df_test])
df_with_predictions.to_csv("dataset_with_predictions.csv", index = False)

