import csv

# Load CSV data
def load_csv(filename):
    with open(filename, 'r') as file:
        data = list(csv.reader(file))
    return data[1:]  # skip header


# Check if hypothesis h is consistent with example x
def consistent(h, x):
    for h_val, x_val in zip(h, x):
        if h_val != '?' and h_val != x_val:
            return False
    return True


# More general check
def more_general(h1, h2):
    return all(h1[i] == '?' or h1[i] == h2[i] for i in range(len(h1)))


# Candidate Elimination Algorithm
def candidate_elimination(data):
    attributes = len(data[0]) - 1

    # Initialize S and G
    S = ['Φ'] * attributes
    G = [['?'] * attributes]

    for example in data:
        x = example[:-1]
        label = example[-1]

        # Positive example
        if label == 'Yes':
            # Remove inconsistent hypotheses from G
            G = [g for g in G if consistent(g, x)]

            # Generalize S
            for i in range(attributes):
                if S[i] == 'Φ':
                    S[i] = x[i]
                elif S[i] != x[i]:
                    S[i] = '?'

        # Negative example
        else:
            new_G = []
            for g in G:
                if consistent(g, x):
                    for i in range(attributes):
                        if g[i] == '?':
                            if S[i] != '?' and S[i] != 'Φ':
                                new_h = g.copy()
                                new_h[i] = S[i]
                                new_G.append(new_h)
                else:
                    new_G.append(g)

            # Remove overly specific hypotheses
            G = [g for g in new_G if more_general(g, S)]

    return S, G


# Run the algorithm
data = load_csv("training_data.csv")
S, G = candidate_elimination(data)

print("Specific Boundary (S):")
print(S)

print("\nGeneral Boundary (G):")
for g in G:
    print(g)
