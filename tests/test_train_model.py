import numpy as np
import pytest

from model.train_model import calculate_kwh


def base_row(**overrides):
    row = {
        "num_lights": 0,
        "num_fans": 0,
        "num_ac": 0,
        "num_people": 0,
        "house_size_sqft": 0,
        "appliance_age_years": 0,
        "has_geyser": 0,
        "has_fridge": 0,
        "city": "Bangalore",
        "building_type": "Apartment",
        "is_metro": 1,
    }
    row.update(overrides)
    return row


def test_floor_is_enforced_for_a_near_empty_household():
    np.random.seed(0)
    # Even with every appliance count at 0, the base load alone can't reach
    # 50 kWh/month, so the max(50, ...) floor in calculate_kwh must kick in.
    assert calculate_kwh(base_row()) == 50


def test_more_appliances_increases_consumption():
    np.random.seed(1)
    low = calculate_kwh(base_row())
    np.random.seed(1)
    high = calculate_kwh(base_row(num_lights=20, num_fans=5, num_ac=2, num_people=4))
    assert high > low


def test_city_multiplier_direction():
    # CITY_MULTIPLIER: Delhi 1.25 vs Bangalore 0.8 - Delhi should predict higher
    # for an otherwise-identical household.
    household = base_row(num_lights=15, num_fans=5, num_ac=1, num_people=3, house_size_sqft=1200)
    np.random.seed(2)
    bangalore = calculate_kwh({**household, "city": "Bangalore"})
    np.random.seed(2)
    delhi = calculate_kwh({**household, "city": "Delhi"})
    assert delhi > bangalore


def test_building_type_factor_direction():
    # BUILDING_FACTOR: Villa 1.3 vs Apartment 1.0.
    household = base_row(num_lights=15, num_fans=5, num_ac=1, num_people=3, house_size_sqft=1200)
    np.random.seed(3)
    apartment = calculate_kwh({**household, "building_type": "Apartment"})
    np.random.seed(3)
    villa = calculate_kwh({**household, "building_type": "Villa"})
    assert villa > apartment


def test_metro_factor_direction():
    # METRO_FACTOR is intentionally inverted: non-metro (0) is 1.05, metro
    # (1) is 1.0 - locking this in since it's easy to assume the opposite.
    household = base_row(num_lights=15, num_fans=5, num_ac=1, num_people=3, house_size_sqft=1200)
    np.random.seed(4)
    metro = calculate_kwh({**household, "is_metro": 1})
    np.random.seed(4)
    non_metro = calculate_kwh({**household, "is_metro": 0})
    assert non_metro > metro


def test_older_appliances_increase_consumption():
    household = base_row(num_lights=15, num_fans=5, num_ac=1, num_people=3, house_size_sqft=1200)
    np.random.seed(5)
    newer = calculate_kwh({**household, "appliance_age_years": 0})
    np.random.seed(5)
    older = calculate_kwh({**household, "appliance_age_years": 10})
    assert older > newer


@pytest.mark.parametrize("seed", range(10))
def test_result_never_goes_below_the_floor(seed):
    np.random.seed(seed)
    assert calculate_kwh(base_row()) >= 50
