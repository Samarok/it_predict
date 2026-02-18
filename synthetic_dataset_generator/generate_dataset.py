import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. НАСТРОЙКИ ГЕНЕРАЦИИ
# ------------------------------------------------------------
N_PROJECTS = 2000        # Количество проектов в датасете
RANDOM_SEED = 42         # Для воспроизводимости
OUTPUT_DIR = "data"      # Папка для сохранения результата
OUTPUT_FILE = "projects_dataset.csv"

# ------------------------------------------------------------
# 2. ФУНКЦИИ ГЕНЕРАЦИИ ОТДЕЛЬНЫХ ПРИЗНАКОВ
# ------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)

def generate_domain(n):
    """Генерирует domain проекта с заданными вероятностями."""
    domains = ['web_development', 'mobile_app', 'enterprise_system', 'data_analytics']
    probs = [0.3, 0.25, 0.25, 0.2]  # в сумме 1.0
    return np.random.choice(domains, size=n, p=probs)

def generate_client_type(n):
    """Генерирует тип заказчика."""
    types = ['government', 'large_corporate', 'sme', 'internal']
    probs = [0.2, 0.4, 0.3, 0.1]
    return np.random.choice(types, size=n, p=probs)

def generate_allocated_time(n):
    """
    Генерирует выделенное время (в неделях).
    Используется логнормальное распределение, чтобы большинство проектов были средними,
    затем значения ограничиваются диапазоном [4, 52].
    """
    raw = np.random.lognormal(mean=2.5, sigma=0.6, size=n)
    times = np.clip(np.round(raw).astype(int), 4, 52)
    return times

def generate_budget_adequacy(n):
    """Генерирует адекватность бюджета (равномерно от 0.5 до 1.5)."""
    return np.random.uniform(0.5, 1.5, size=n)

def generate_team_size_adequacy(n):
    """Генерирует адекватность размера команды (равномерно от 0.6 до 2.0)."""
    return np.random.uniform(0.6, 2.0, size=n)

def generate_methodology(n):
    """Генерирует методологию разработки."""
    methods = ['agile', 'waterfall', 'hybrid']
    probs = [0.5, 0.2, 0.3]
    return np.random.choice(methods, size=n, p=probs)

def generate_tz_quality(n):
    """Генерирует качество ТЗ (равномерно от 0.2 до 1.0)."""
    return np.random.uniform(0.2, 1.0, size=n)

def generate_stakeholder_involvement(n):
    """Генерирует вовлеченность стейкхолдеров (равномерно от 0.1 до 1.0)."""
    return np.random.uniform(0.1, 1.0, size=n)

def generate_risk_skill_gap(n):
    """Генерирует риск недостаточной квалификации (True/False) с вероятностью 0.3."""
    return np.random.choice([True, False], size=n, p=[0.3, 0.7])

# ------------------------------------------------------------
# 3. РАСЧЁТ ЦЕЛЕВЫХ ПЕРЕМЕННЫХ
# ------------------------------------------------------------
def calculate_actual_duration(row):
    """
    Для одного проекта (строка DataFrame) вычисляет фактическую длительность.
    Использует все признаки, включая категориальные, для расчёта риска.
    """
    allocated = row['allocated_time']

    # Базовый риск на основе количественных признаков
    base_risk = (
        (1 - row['budget_adequacy']) * 0.3 +
        (1 - row['team_size_adequacy']) * 0.2 +
        (1 - row['tz_quality']) * 0.25 +
        (1 - row['stakeholder_involvement']) * 0.15 +
        (0.1 if row['risk_skill_gap'] else 0)
    )

    # Модификаторы в зависимости от категорий
    if row['domain'] == 'enterprise_system':
        base_risk += 0.15
    if row['client_type'] == 'government':
        base_risk += 0.2
    if row['methodology'] == 'waterfall' and row['tz_quality'] < 0.5:
        base_risk += 0.1

    # Случайный фактор, чтобы добавить вариативности
    random_factor = np.random.uniform(0.5, 1.5)

    # Итоговая длительность (не может быть меньше 50% от плана)
    actual = allocated * (1 + base_risk * random_factor)
    actual = max(actual, allocated * 0.5)
    return int(round(actual))

# ------------------------------------------------------------
# 4. ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ДАТАСЕТА
# ------------------------------------------------------------
def generate_projects(n, seed=RANDOM_SEED):
    set_seed(seed)

    # Генерация всех признаков
    domains = generate_domain(n)
    client_types = generate_client_type(n)
    allocated_times = generate_allocated_time(n)
    budget_adequacy = generate_budget_adequacy(n)
    team_size_adequacy = generate_team_size_adequacy(n)
    methodologies = generate_methodology(n)
    tz_quality = generate_tz_quality(n)
    stakeholder_involvement = generate_stakeholder_involvement(n)
    risk_skill_gap = generate_risk_skill_gap(n)

    # Сборка DataFrame
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

    # Вычисляем actual_duration для каждого проекта
    df['actual_duration'] = df.apply(calculate_actual_duration, axis=1)

    # Целевые переменные
    df['delay_percentage'] = ((df['actual_duration'] / df['allocated_time']) - 1) * 100
    df['completed_on_time'] = df['delay_percentage'] <= 10.0

    return df

# ------------------------------------------------------------
# 5. ЗАПУСК ГЕНЕРАЦИИ И СОХРАНЕНИЕ
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Генерация {N_PROJECTS} проектов с seed={RANDOM_SEED}...")
    df_projects = generate_projects(N_PROJECTS, RANDOM_SEED)

    # Создание папки, если её нет
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    # Сохранение в CSV
    df_projects.to_csv(output_path, index=False)
    print(f"Датасет сохранён: {output_path}")

    # Краткая статистика
    print("\nПервые 5 записей:")
    print(df_projects.head())

    print("\nСтатистика по целевым переменным:")
    print(f"Доля проектов, завершённых в срок: {df_projects['completed_on_time'].mean():.2%}")
    print(f"Средняя задержка: {df_projects['delay_percentage'].mean():.2f}%")
    print(f"Медианная задержка: {df_projects['delay_percentage'].median():.2f}%")
    print(f"Минимальная/максимальная длительность проекта (недель): {df_projects['actual_duration'].min()} / {df_projects['actual_duration'].max()}")