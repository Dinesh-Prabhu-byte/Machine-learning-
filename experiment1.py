# FIND-S Algorithm Implementation

def find_s(training_data):
    # Initialize hypothesis with most specific values
    hypothesis = ['Φ'] * (len(training_data[0]) - 1)

    for instance in training_data:
        # Consider only positive examples
        if instance[-1] == 'Yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] == 'Φ':
                    hypothesis[i] = instance[i]
                elif hypothesis[i] != instance[i]:
                    hypothesis[i] = '?'

    return hypothesis


# Training data
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No']
]

# Run FIND-S
final_hypothesis = find_s(training_data)

print("Most Specific Hypothesis:")
print(final_hypothesis)
