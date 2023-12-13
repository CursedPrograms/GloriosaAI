import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Create Antecedent/Consequent objects representing linguistic variables
color = ctrl.Antecedent(np.arange(0, 256, 1), 'color')
shape = ctrl.Antecedent(np.arange(0, 101, 1), 'shape')
artistic_value = ctrl.Consequent(np.arange(0, 101, 1), 'artistic_value')

# Generate fuzzy membership functions for linguistic variables
color['low'] = fuzz.trimf(color.universe, [0, 0, 128])
color['medium'] = fuzz.trimf(color.universe, [0, 128, 255])
color['high'] = fuzz.trimf(color.universe, [128, 255, 255])

shape['small'] = fuzz.trimf(shape.universe, [0, 0, 50])
shape['medium'] = fuzz.trimf(shape.universe, [0, 50, 100])
shape['large'] = fuzz.trimf(shape.universe, [50, 100, 100])

artistic_value['low'] = fuzz.trimf(artistic_value.universe, [0, 0, 50])
artistic_value['medium'] = fuzz.trimf(artistic_value.universe, [0, 50, 100])
artistic_value['high'] = fuzz.trimf(artistic_value.universe, [50, 100, 100])

# Define rules for the Fuzzy Logic System
rule1 = ctrl.Rule(color['low'] & shape['small'], artistic_value['low'])
rule2 = ctrl.Rule(color['medium'] & shape['medium'], artistic_value['medium'])
rule3 = ctrl.Rule(color['high'] & shape['large'], artistic_value['high'])

# Create the Fuzzy Control System
art_system = ctrl.ControlSystem([rule1, rule2, rule3])
art_simulator = ctrl.ControlSystemSimulation(art_system)

# Generate art based on input values
def generate_art(color_value, shape_value):
    # Pass input values to the Fuzzy Logic System
    art_simulator.input['color'] = color_value
    art_simulator.input['shape'] = shape_value

    # Compute the output (artistic value)
    art_simulator.compute()

    # Return the result
    return art_simulator.output['artistic_value']

# Example usage
color_value = 150
shape_value = 75

result = generate_art(color_value, shape_value)
print(f'Artistic Value: {result}')

# Visualize fuzzy sets
color.view()
shape.view()
artistic_value.view()
plt.show()