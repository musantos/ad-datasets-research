import torch
import torch.nn as nn

class SimpleTrajectoryPredictor(nn.Module):
    def __init__(self, input_steps=11, output_steps=80):
        super(SimpleTrajectoryPredictor, self).__init__()
        
        # Entrada: 11 frames * 2 coordenadas (x,y) = 22 numeros
        self.input_dim = input_steps * 2
        # Saida: 80 frames * 2 coordenadas (x,y) = 160 numeros
        self.output_dim = output_steps * 2
        
        # Uma rede simples com 2 camadas escondidas
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, x):
        # x chega como [batch, 11, 2]
        batch_size = x.shape[0]
        
        # Achatar para [batch, 22]
        x = x.view(batch_size, -1)
        
        # Passar pela rede
        prediction = self.network(x)
        
        # Voltar para o formato [batch, 80, 2]
        return prediction.view(batch_size, 80, 2)

if __name__ == "__main__":
    model = SimpleTrajectoryPredictor()
    print("OK: Modelo de Predicao carregado.")
    # Teste de shape
    test_input = torch.randn(2, 11, 2)
    output = model(test_input)
    print(f"Shape da Predicao: {output.shape}") # Esperado: [2, 80, 2]
