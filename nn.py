import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def train_nn(X_train, X_val_in, y_train, y_val_in, X_test_in):
    # Use the GPU if possible
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("### GPU is available and being used ###")
    else:
        device = torch.device("cpu")
        print("### GPU is not available, using CPU instead ###")

    # Convert the input data into tensors
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val = torch.tensor(X_val_in, dtype=torch.float32)
    y_val = torch.tensor(y_val_in, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test_in, dtype=torch.float32)

    # Move tensors to the GPU
    X = X.to(device)
    y = y.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    
    # Define the model
    model = nn.Sequential(
        nn.Linear(20, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )
    model = model.to(device)

    # Define the loss function and optimizer
    loss_fn = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # Training the model
    print("### Training ###")
    N_EPOCHS = 10000
    BATCH_SIZE = 65536

    for epoch in range(N_EPOCHS):
        for i in range(0, len(X), BATCH_SIZE):
            Xbatch = X[i:i+BATCH_SIZE]
            y_pred = model(Xbatch)
            ybatch = y[i:i+BATCH_SIZE]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')

    # Compute Accuracy
    with torch.no_grad():
        y_pred = model(X_val)

    accuracy = (y_pred.round() == y_val).float().mean()
    print(f"Accuracy {accuracy}")

    # Output predictions
    with torch.no_grad():
        y_out = model(X_test)

    y_out = y_out.cpu()

    y_out_df = pd.DataFrame(y_out.numpy(), columns=['TX_FRAUD'])
    y_out_df.insert(0, 'TX_ID', pd.read_csv("data/transactions_test.csv")['TX_ID'])
    y_out_df.to_csv('data/submission.csv', index=False)