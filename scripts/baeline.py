import json
import math

def check(sub, theta):
    with open(f'25_math_{sub}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    alpha = 3.1
    beta = 5.4
    gamma = 1.7
    tau = 150
    total_time = 6000

    problems = data['problems']
    total_difficulty = sum(p['difficulty'] for p in problems)

    def calculate_entropy(choice_rate):
        entropy = 0
        for k, v in choice_rate.items():
            if k != 'correct' and v > 0:
                entropy -= v * math.log(v)
        return entropy

    for p in problems:
        p['t_i'] = total_time * (p['difficulty'] / total_difficulty)
        p['d_i'] = p['difficulty']
        p['c_i'] = 0.2 if p['problem_type'] == 'objective' else 0.0
        if p['problem_type'] == 'objective':
            p['a_i'] = calculate_entropy(p['choice_rate'])
        else:
            p['a_i'] = 0.0

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    expected_score = 0
    for p in problems:
        term_time = alpha * math.log(1 + p['t_i'] / tau)
        X = theta - beta * p['d_i'] - gamma * p['a_i'] + term_time
        prob_correct = p['c_i'] + (1 - p['c_i']) * sigmoid(X)
        expected_score += prob_correct * p['score']
        
    print(f"Sub {sub} : theta={theta} -> score={expected_score:.2f}")

for theta in [1,2,3]:
    for sub in ["calculus","geometry","prob_stat"]:
        check(sub=sub,theta=theta)