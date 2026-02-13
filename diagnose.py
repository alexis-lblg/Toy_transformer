from data import Data
import torch

def diagnose():
    print("=== DIAGNOSTIC DU DATASET ===")
    
    # Charger les données
    tokens, labels, tokens_t, labels_t = torch.load("dataset.pt")
    
    print(f"Shape train tokens: {tokens.shape}")
    print(f"Shape test tokens: {tokens_t.shape}")
    print(f"Token range train: {tokens.min().item()} - {tokens.max().item()}")
    print(f"Token range test: {tokens_t.min().item()} - {tokens_t.max().item()}")
    
    # Vérifier la distribution des fonctions
    print(f"\nDistribution des fonctions (train):")
    print(f"  f0: {(tokens[:, 1] == 0).sum().item()} échantillons")
    print(f"  f1: {(tokens[:, 1] == 1).sum().item()} échantillons")
    
    print(f"\nDistribution des fonctions (test):")
    print(f"  f0: {(tokens_t[:, 1] == 0).sum().item()} échantillons")
    print(f"  f1: {(tokens_t[:, 1] == 1).sum().item()} échantillons")
    
    # Vérifier les labels
    print(f"\nRange des labels (train): {labels.min().item()} - {labels.max().item()}")
    print(f"Range des labels (test): {labels_t.min().item()} - {labels_t.max().item()}")
    
    # Tester un modèle très simple
    print("\n=== TEST DE BASELINE ===")
    model = Data()
    
    # Performance avant entraînement
    print("Performance avant entraînement:")
    val_loss, val_acc = model.test()
    print(f"  Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")
    
    # Quelques steps d'entraînement rapide
    print("\nEntraînement rapide (1000 steps)...")
    for step in range(1000):
        logits = model.model(model.tokens)[:, -1, :]
        loss = F.cross_entropy(logits, model.labels)
        
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        
        if step % 200 == 0:
            train_acc = (logits.argmax(dim=-1) == model.labels).float().mean().item()
            val_loss, val_acc = model.test()
            print(f"  Step {step}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

