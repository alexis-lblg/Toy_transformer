import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch.optim as optim
import matplotlib.pyplot as plt

class Data:
    def __init__(self, weight_decay=0.0001):
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokens, self.labels, self.tokens_t, self.labels_t = torch.load("dataset.pt")

        self.tokens = self.tokens.to(self.device)
        self.labels = self.labels.to(self.device)
        self.tokens_t = self.tokens_t.to(self.device)
        self.labels_t = self.labels_t.to(self.device)


        self.K = 2
        self.N_X = 1000
        self.N_Y = 1000
        
        self.d_vocab = 2 + self.N_X  
        self.seq_len = 2

        cfg = HookedTransformerConfig(
            n_layers=6,
            n_heads=8,
            d_model=256,
            d_head=32,
            d_mlp=512,
            d_vocab=self.d_vocab,
            n_ctx=self.seq_len,
            act_fn="relu",
            normalization_type=None,
            d_vocab_out=self.N_Y,
            device=self.device
        )
        self.model = HookedTransformer(cfg)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=weight_decay) #With weight
        
        self.train_losses = []
        self.val_losses = []
        self.weight_norms = []  

    def run(self, num_steps=20000):
        
        for step in range(num_steps):
            logits = self.model(self.tokens)[:, -1, :]
            
            ce_loss = F.cross_entropy(logits, self.labels)
            
            #
            weight_norm = 0
            for param in self.model.parameters():
                weight_norm += param.norm().item()
            
            total_loss = ce_loss 
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            self.train_losses.append(ce_loss.item())
            self.weight_norms.append(weight_norm)
            
            if step % 200 == 0:
                val_loss, val_acc = self.test()
                self.val_losses.append(val_loss)
                
                train_acc = (logits.argmax(dim=-1) == self.labels).float().mean().item()
                print(f"step {step:4d} | train loss {ce_loss.item():.4f} | train acc {train_acc:.3f} | "
                      f"val loss {val_loss:.4f} | val acc {val_acc:.3f} | weight norm {weight_norm:.2f}")
                print(logits.abs().max().item())
        
        self.plot_training()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.tokens_t)[:, -1, :]
            loss = F.cross_entropy(logits, self.labels_t)
            preds = logits.argmax(dim=-1)
            acc = (preds == self.labels_t).float().mean().item()
        self.model.train()
        return loss.item(), acc

    def plot_training(self):
        if not self.train_losses:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            val_steps = [i*200 for i in range(len(self.val_losses))]
            axes[0].plot(val_steps, self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss during the training')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.weight_norms)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Norm L2 of the weights')
        axes[1].set_title('Evolution of the weights')
        axes[1].grid(True)
        
        self.plot_embeddings(ax=axes[2])
        
        plt.tight_layout()
        plt.show()

    def plot_embeddings(self, ax=None):
        W_E = self.model.W_E.detach()
        E_f = W_E[:2]  
        E_x = W_E[2:1002]  
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        scatter = ax.scatter(E_x[:, 0], E_x[:, 1], s=5, alpha=0.5, c=range(1000), cmap="hsv")
        
        ax.scatter(E_f[0, 0], E_f[0, 1], s=100, marker='*', color='red', label='f0')
        ax.scatter(E_f[1, 0], E_f[1, 1], s=100, marker='*', color='blue', label='f1')
        
        ax.set_title("Embeddings space")
        ax.set_xlabel("Dimension 0")
        ax.set_ylabel("Dimension 1")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax

    def evaluate(self):
        
        val_loss, val_acc = self.test()
        print(f"Test complet: loss={val_loss:.4f}, acc={val_acc:.3f}")
        
        self.model.eval()
        with torch.no_grad():
            for k in range(self.K):
                mask = self.tokens_t[:, 1] == k
                if mask.any():
                    logits_k = self.model(self.tokens_t[mask])[:, -1, :]
                    acc_k = (logits_k.argmax(dim=-1) == self.labels_t[mask]).float().mean().item()
                    print(f"  f{k}: accuracy={acc_k:.3f}")
        
        self.model.train()
        return val_acc