from src.engine import objective_weights, traffic_multiplier, scheduling_class


def test_objective_weights_has_carbon():
    weights = objective_weights("Carbon")
    assert weights["carbon"] > weights["cost"]


def test_traffic_multiplier_spiky_greater_than_steady():
    assert traffic_multiplier("Spiky") > traffic_multiplier("Steady")


def test_scheduling_class_edge():
    assert scheduling_class("Jetson AGX Xavier") == "Edge"
