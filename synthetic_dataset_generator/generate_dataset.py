import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. GENERATION SETTINGS
# ------------------------------------------------------------
N_PROJECTS = 2000        # Number of projects in the dataset
RANDOM_SEED = 42         # For reproducibility
OUTPUT_DIR = "data"      # Folder for saving results
OUTPUT_FILE = "projects_dataset.csv"

# ------------------------------------------------------------
# 2. FEATURE GENERATION FUNCTIONS
# ------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)

def generate_domain(n):
    """Generates the project domain with specified probabilities."""
    domains = ['web_development', 'mobile_app', 'enterprise_system', 'data_analytics']
    probs = [0.3, 0.25, 0.25, 0.2]  # sum to 1.0
    return np.random.choice(domains, size=n, p=probs)

def generate_client_type(n):
    """Generates the client type."""
    types = ['government', 'large_corporate', 'sme', 'internal']
    probs = [0.2, 0.4, 0.3, 0.1]
    return np.random.choice(types, size=n, p=probs)

def generate_allocated_time(n):
    """
    Generates allocated time (in weeks).
    Uses a log-normal distribution so that most projects are average,
    then values are clipped to the range [4, 52].
    """
    raw = np.random.lognormal(mean=2.5, sigma=0.6, size=n)
    times = np.clip(np.round(raw).astype(int), 4, 52)
    return times

def generate_budget_adequacy(n):
    """Generates budget adequacy (uniformly from 0.5 to 1.5)."""
    return np.random.uniform(0.5, 1.5, size=n)

def generate_team_size_adequacy(n):
    """Generates team size adequacy (uniformly from 0.6 to 2.0)."""
    return np.random.uniform(0.6, 2.0, size=n)

def generate_methodology(n):
    """Generates the development methodology."""
    methods = ['agile', 'waterfall', 'hybrid']
    probs = [0.5, 0.2, 0.3]
    return np.random.choice(methods, size=n, p=probs)

def generate_tz_quality(n):
    """Generates TZ (technical specification) quality (uniformly from 0.2 to 1.0)."""
    return np.random.uniform(0.2, 1.0, size=n)

def generate_stakeholder_involvement(n):
    """Generates stakeholder involvement (uniformly from 0.1 to 1.0)."""
    return np.random.uniform(0.1, 1.0, size=n)

def generate_risk_skill_gap(n):
    """Generates skill gap risk (True/False) with probability 0.3."""
    return np.random.choice([True, False], size=n, p=[0.3, 0.7])

# ------------------------------------------------------------
# 3. TARGET VARIABLE CALCULATION
# ------------------------------------------------------------
def calculate_actual_duration(row):
    """
    Improved version: nonlinear effects, feature interactions,
    heteroscedasticity (noise depends on planning quality).
    """
    allocated = row['allocated_time']

    # ============================================================
    # 1. Base deficits (nonlinear transformations)
    # ============================================================

    # Budget deficit: quadratic dependence
    budget_deficit = max(0, 1 - row['budget_adequacy'])
    budget_surplus = max(0, row['budget_adequacy'] - 1)
    budget_impact = (budget_deficit ** 1.5) * 0.4 - (budget_surplus ** 0.7) * 0.1

    # Team deficit: threshold effect
    team_ratio = row['team_size_adequacy']
    if team_ratio < 0.6:
        team_impact = (0.6 - team_ratio) * 0.8
    elif team_ratio < 0.9:
        team_impact = (0.9 - team_ratio) * 0.3
    else:
        team_impact = -min(0, (team_ratio - 1.2)) * 0.05

    # TZ quality: exponential effect
    tz_impact = np.exp(-row['tz_quality'] * 2) * 0.35 - 0.05

    # Stakeholder involvement: logistic effect (higher involvement = lower risk)
    stakeholder_impact = (0.5 - 1 / (1 + np.exp(8 * (0.5 - row['stakeholder_involvement'])))) * 0.25

    # Skill gap: amplifies other risks
    skill_gap_impact = 0.08 if row['risk_skill_gap'] else 0
    skill_gap_multiplier = 1.3 if row['risk_skill_gap'] else 1.0

    # ============================================================
    # 2. Feature interactions (cross-terms)
    # ============================================================

    # Poor TZ + uninvolved stakeholders
    tz_stakeholder_interaction = (1 - row['tz_quality']) * (1 - row['stakeholder_involvement']) * 0.2

    # Budget deficit + team deficit
    budget_team_interaction = budget_deficit * max(0, 1 - team_ratio) * 0.25

    # Low skill + poor TZ
    skill_tz_interaction = (0.15 if row['risk_skill_gap'] else 0) * (1 - row['tz_quality'])

    # ============================================================
    # 3. Categorical feature modifiers
    # ============================================================

    domain_multiplier = {
        'web_development': 0.9,
        'mobile_app': 1.0,
        'enterprise_system': 1.4,
        'data_analytics': 1.1
    }.get(row['domain'], 1.0)

    client_multiplier = {
        'government': 1.5,
        'large_corporate': 1.2,
        'sme': 0.9,
        'internal': 0.8
    }.get(row['client_type'], 1.0)

    methodology_penalty = 0
    if row['methodology'] == 'waterfall' and row['tz_quality'] < 0.5:
        methodology_penalty = 0.25
    elif row['methodology'] == 'agile':
        methodology_penalty = -0.05

    # ============================================================
    # 4. Base risk summation
    # ============================================================

    base_risk = (
        budget_impact +
        team_impact +
        tz_impact +
        stakeholder_impact +
        skill_gap_impact +
        tz_stakeholder_interaction +
        budget_team_interaction +
        skill_tz_interaction +
        methodology_penalty
    )

    base_risk = base_risk * domain_multiplier * client_multiplier * skill_gap_multiplier
    base_risk = np.clip(base_risk, -0.2, 1.2)

    # ============================================================
    # 5. Noise: heteroscedastic
    # ============================================================

    uncertainty = 0.3 + (1 - row['tz_quality']) * 0.7

    if np.random.random() < 0.7:
        # Lognormal noise (heavy tail towards delays)
        random_factor = np.random.lognormal(mean=0, sigma=uncertainty * 0.4)
    else:
        # Normal noise (can also be acceleration)
        random_factor = 1 + np.random.normal(0, uncertainty * 0.15)

    random_factor = np.clip(random_factor, 0.4, 2.0)

    # ============================================================
    # 6. Final duration
    # ============================================================

    actual = allocated * (1 + base_risk) * random_factor
    actual = np.clip(actual, allocated * 0.4, allocated * 2.5)

    return int(round(actual))

# ------------------------------------------------------------
# 4. MAIN DATASET GENERATION FUNCTION
# ------------------------------------------------------------
def generate_projects(n, seed=RANDOM_SEED):
    set_seed(seed)

    # Generate all features
    domains = generate_domain(n)
    client_types = generate_client_type(n)
    allocated_times = generate_allocated_time(n)
    budget_adequacy = generate_budget_adequacy(n)
    team_size_adequacy = generate_team_size_adequacy(n)
    methodologies = generate_methodology(n)
    tz_quality = generate_tz_quality(n)
    stakeholder_involvement = generate_stakeholder_involvement(n)
    risk_skill_gap = generate_risk_skill_gap(n)

    # Build DataFrame
    df = pd.DataFrame({
        'project_id': range(1, n + 1),
        'domain': domains,
        'client_type': client_types,
        'allocated_time': allocated_times,
        'budget_adequacy': budget_adequacy,
        'team_size_adequacy': team_size_adequacy,
        'methodology': methodologies,
        'tz_quality': tz_quality,
        'stakeholder_involvement': stakeholder_involvement,
        'risk_skill_gap': risk_skill_gap
    })

    # Calculate actual_duration for each project
    df['actual_duration'] = df.apply(calculate_actual_duration, axis=1)

    # Target variables
    df['delay_percentage'] = ((df['actual_duration'] / df['allocated_time']) - 1) * 100
    df['completed_on_time'] = df['delay_percentage'] <= 10.0

    return df

# ------------------------------------------------------------
# 5. RUN GENERATION AND SAVE
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Generating {N_PROJECTS} projects with seed={RANDOM_SEED}...")
    df_projects = generate_projects(N_PROJECTS, RANDOM_SEED)

    # Create folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    # Save to CSV
    df_projects.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path}")

    # Brief statistics
    print("\nFirst 5 records:")
    print(df_projects.head())

    print("\nTarget variable statistics:")
    print(f"Percentage of projects completed on time: {df_projects['completed_on_time'].mean():.2%}")
    print(f"Average delay: {df_projects['delay_percentage'].mean():.2f}%")
    print(f"Median delay: {df_projects['delay_percentage'].median():.2f}%")
    print(f"Min/Max project duration (weeks): {df_projects['actual_duration'].min()} / {df_projects['actual_duration'].max()}")