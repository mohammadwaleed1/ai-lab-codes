from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the structure of the Bayesian Network
model = DiscreteBayesianNetwork([
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
])



# P(Burglary) - [False, True]
cpd_burglary = TabularCPD(variable='Burglary', variable_card=2,values=[[0.999], [0.001]])

# P(Earthquake) - [False, True]
cpd_earthquake = TabularCPD(variable='Earthquake', variable_card=2, values=[[0.998], [0.002]])

#
cpd_alarm = TabularCPD(
    variable='Alarm',variable_card=2,
    values=[
        [0.999, 0.71, 0.06, 0.05], # Alarm = False
        [0.001, 0.29, 0.94, 0.95]  # Alarm = True
    ],
    evidence=['Burglary', 'Earthquake'],
    evidence_card=[2, 2]
)

# P(JohnCalls | Alarm)
cpd_john = TabularCPD(
    variable='JohnCalls',variable_card=2,
    values=[
        [0.95, 0.1], # JohnCalls = False
        [0.05, 0.9]  # JohnCalls = True
    ],
    evidence=['Alarm'],
    evidence_card=[2]
)

# P(MaryCalls | Alarm)
cpd_mary = TabularCPD(
    variable='MaryCalls',
    variable_card=2,
    values=[
        [0.99, 0.3], # MaryCalls = False
        [0.01, 0.7]  # MaryCalls = True
    ],
    evidence=['Alarm'],
    evidence_card=[2]
)

# Step 3: Add CPDs to the model
model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_john, cpd_mary)

# Step 4: Verify the model
# check_model ensures the logic is sound and probabilities sum to 1
assert model.check_model(), "Model is incorrect"

# Step 5: Perform inference
inference = VariableElimination(model)

print("Querying Probability of Burglary state given Alarm=True...")
result = inference.query(variables=['Burglary'], evidence={'MaryCalls': 1,'JohnCalls': 1})

print(result)
