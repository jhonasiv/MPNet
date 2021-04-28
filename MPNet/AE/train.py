import torch
from torch.autograd import Variable
from torch.nn import MSELoss

import data_loader as dl
import argparse
import os
from CAE import Encoder, Decoder
from CAE import loss_function
from plotly import graph_objs as go

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def plot(inp, output, title):
    inp = inp.cpu().detach().numpy().reshape((-1, 2))
    output = output.cpu().detach().numpy().reshape((-1, 2))
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=inp[:, 0], y=inp[:, 1], mode='markers', marker={'color': 'black', 'size': 5}))
    fig.add_trace(go.Scatter(x=output[:, 0], y=output[:, 1], mode='markers', marker={'color': 'blue', 'size': 5}))
    fig.update_xaxes(range=[-20, 20])
    fig.update_yaxes(range=[-20, 20])
    fig.update_layout(title=title)
    fig.show()


def main(args):
    if not os.path.exists(f"{project_path}/{args.model_path}"):
        os.makedirs(f"{project_path}/{args.model_path}")
    
    training_data = dl.loader(50000, 100)
    validation_data = dl.loader(7500, 1, 50000)
    
    encoder = Encoder()
    decoder = Decoder()
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    
    mse_loss = MSELoss()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    total_loss = []
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}/{args.num_epochs}:")
        avg_loss = 0
        for batch, inp in enumerate(training_data):
            optimizer.zero_grad()
            decoder.zero_grad()
            encoder.zero_grad()
            inp = Variable(inp).cuda().float()
            # ===================forward=====================
            h = encoder(inp)
            output = decoder(h)
            keys = encoder.state_dict().keys()
            weight = encoder.state_dict()['encoder.6.weight']
            loss = loss_function(weight, inp, output, h)
            avg_loss += loss.data[0]
            if batch == 100 and (epoch + 1) % 20 == 0:
                plot(inp[50], output[50], 'Training')
            # ===================backward====================
            loss.mean().backward()
            optimizer.step()
        
        avg_val_loss = 0
        for n, val_inp in enumerate(validation_data):
            val_inp = Variable(val_inp).cuda().float()
            # ===================forward=====================
            output = encoder(val_inp)
            output = decoder(output)
            loss = mse_loss(output, val_inp)
            avg_val_loss += loss.item()
            if n == 100 and (epoch + 1) % 20 == 0:
                plot(val_inp, output, 'Validation')
        # ===================backward====================
        print(f"\t Average loss\t {avg_loss / len(training_data) / args.batch_size}\t Val loss\t "
              f"{avg_val_loss / 7500}")
        total_loss.append(avg_loss / (len(training_data) / args.batch_size))
    
    torch.save(encoder.state_dict(), f"{project_path}/{args.model_path}/cae_encoder.pt")
    torch.save(decoder.state_dict(), f"{project_path}/{args.model_path}/cae_decoder.pt")
    torch.save(total_loss, 'total_loss.dat')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=60)
    parser.add_argument('--model_path', default="models")
    parser.add_argument('--batch_size', default=100)
    
    args = parser.parse_args()
    
    main(args)
