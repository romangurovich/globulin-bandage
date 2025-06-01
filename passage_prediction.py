import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Step 1: Add membrane passage prediction
def predict_membrane_passage(patch_sequence):
    """Predict likelihood of patch passing through RBC membrane"""
    # Simple hydrophobicity-based model (in reality would use more sophisticated features)
    hydrophobicity = {'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 
                      'F': 1.19, 'G': 0.48, 'H': -0.40, 'I': 1.38,
                      'K': -1.50, 'L': 1.06, 'M': 0.64, 'N': -0.78,
                      'P': 0.12, 'Q': -0.85, 'R': -2.53, 'S': -0.18,
                      'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26}
    
    avg_hydrophobicity = sum(hydrophobicity[aa] for aa in patch_sequence)/len(patch_sequence)
    # Simple threshold model (would train a proper classifier in practice)
    membrane_passage_prob = 1 / (1 + np.exp(0.5*(avg_hydrophobicity - 0.3)))
    return membrane_passage_prob

# Step 2: Add toxicity prediction
def predict_toxicity(patch_sequence):
    """Predict likelihood of patch reacting with off-target sites"""
    # Simple charge-based model (would use more features in reality)
    positive_aas = ['K', 'R', 'H']
    negative_aas = ['D', 'E']
    net_charge = sum(1 for aa in patch_sequence if aa in positive_aas)
    net_charge -= sum(1 for aa in patch_sequence if aa in negative_aas)
    
    # Simple logistic model (would train properly in practice)
    toxicity_prob = 1 / (1 + np.exp(-0.3*(abs(net_charge) - 1)))
    return toxicity_prob

# Step 3: Simulation visualization of patch action
def create_hb_simulation(patch_sequence="KSAVTALWG"):
    """Create animation showing patch disrupting HbS polymerization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Hydrophilic Patch Disrupting HbS Polymerization")
    ax.axis('off')
    
    # Create hemoglobin molecules (circles)
    hb1 = plt.Circle((3, 5), 0.8, color='red', alpha=0.7)
    hb2 = plt.Circle((5, 5), 0.8, color='red', alpha=0.7)
    hb3 = plt.Circle((7, 5), 0.8, color='red', alpha=0.7)
    ax.add_patch(hb1)
    ax.add_patch(hb2)
    ax.add_patch(hb3)
    
    # Create water molecules (blue dots)
    waters = [plt.Circle((np.random.uniform(2,8), np.random.uniform(3,7)), 
              0.1, color='blue', alpha=0.3) for _ in range(30)]
    for water in waters:
        ax.add_patch(water)
    
    # Create patch (initially not visible)
    patch = plt.Circle((1, 5), 0.5, color='green', alpha=0)
    patch_text = ax.text(1, 5, patch_sequence, ha='center', va='center', 
                        color='white', fontsize=8, alpha=0)
    ax.add_patch(patch)
    
    # Animation function
    def update(frame):
        nonlocal patch, patch_text
        
        if frame < 10:
            # Patch enters
            patch.center = (1 + frame*0.3, 5)
            patch.alpha = frame*0.1
            patch_text.set_position((1 + frame*0.3, 5))
            patch_text.set_alpha(frame*0.1)
            
        elif 10 <= frame < 30:
            # Patch binds and pushes water
            if frame == 10:
                patch.set_color('yellow')  # Change color when bound
            
            # Push water molecules away
            for water in waters:
                x, y = water.center
                dx = x - 5
                dy = y - 5
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    new_x = x + 0.02*dx/dist
                    new_y = y + 0.02*dy/dist
                    water.center = (new_x, new_y)
            
            # Separate hemoglobin molecules
            hb1.center = (3 - 0.02*(frame-10), 5)
            hb3.center = (7 + 0.02*(frame-10), 5)
            
        return [patch, patch_text] + waters + [hb1, hb2, hb3]
    
    anim = FuncAnimation(fig, update, frames=40, interval=100, blit=True)
    plt.close()
    return HTML(anim.to_jshtml())

# Example usage with the KSAVTALWG patch
patch_sequence = "KSAVTALWG"

# Predict membrane passage
membrane_prob = predict_membrane_passage(patch_sequence)
print(f"Probability of passing through RBC membrane: {membrane_prob:.2f}")

# Predict toxicity
toxicity_prob = predict_toxicity(patch_sequence)
print(f"Probability of off-target toxicity: {toxicity_prob:.2f}")

# Create and display simulation
simulation = create_hb_simulation(patch_sequence)
simulation
