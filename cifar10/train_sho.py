import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

def train_iteration(net, inputs, targets, optimizer):
    inputs, targets = inputs.cuda(), targets.cuda()
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return loss, correct, total

def train_epoch(net, optimizer, trainloader):
    train_loss = 0 
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        loss, correct_, total_ = train_iteration(net, inputs, targets, optimizer)
        train_loss += loss.item()
        correct += correct_
        total += total_
    return train_loss, correct, total, batch_idx

def test_iteration(net, inputs, targets):
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    return loss, correct, total

def test_epoch(net, testloader):
    test_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            loss, correct_, total_ = test_iteration(net, inputs, targets)
            test_loss += loss.item()
            total += total_
            correct += correct_
    return test_loss, correct, total, batch_idx
    
def train(n_epochs, net, trainloader,testloader, optimizer,scheduler,writer = None):
    for epoch in range(n_epochs):  
        train_loss, correct_train, total_train, batch_idx_train = train_epoch(net.train(), optimizer, trainloader)
        test_loss, correct_test, total_test, batch_idx_test = test_epoch(net.eval(), testloader)
        if scheduler != None:
            scheduler.step()
            
        print('\nEpoch: %d' % epoch)
        print("Train Loss: ", (train_loss/(batch_idx_train+1)), "Accuracy: ", 100.*correct_train/total_train)
        print("Test Loss: ", (test_loss/(batch_idx_test+1)), "Accuracy: ", 100.*correct_test/total_test)
        
        if writer != None:
            writer.add_scalar('Loss/train', (train_loss/(batch_idx_train+1)), epoch)
            writer.add_scalar('Loss/test', (test_loss/(batch_idx_test+1)), epoch)
            writer.add_scalar('Accuracy/train', 100.*correct_train/total_train, epoch)
            writer.add_scalar('Accuracy/test', 100.*correct_test/total_test, epoch)