import numpy as np

from app import DA, DASE, leontief_utilities, maximal_storage_path, validate_instance


def test_small_paper_example() -> None:
    supply = np.array([1.5, 1.0, 0.0])
    demands = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    capacity = 1.0

    validate_instance(demands, supply, capacity)
    da, fair_shares = DA(demands, supply, capacity)
    dase, history = DASE(demands, supply, capacity, da)

    np.testing.assert_allclose(fair_shares, [0.75, 0.5], atol=1e-7)
    np.testing.assert_allclose(
        da,
        [[0.75, 0.0, 0.0], [0.0, 0.0, 0.5]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        dase,
        [[1.5, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-7,
    )
    assert len(history) == 2


def test_four_by_four_finite_capacity_example() -> None:
    supply = np.array([16.0, 0.0, 84.0, 0.0])
    demands = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ]
    )
    capacity = 12.0

    validate_instance(demands, supply, capacity)
    da, _ = DA(demands, supply, capacity)
    dase, _ = DASE(demands, supply, capacity, da)

    expected_da = np.array(
        [
            [4.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 3.0, 3.0],
            [2.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
        ]
    )
    expected_dase = np.array(
        [
            [20.0 / 3.0, 0.0, 20.0 / 3.0, 0.0],
            [0.0, 0.0, 5.0, 5.0],
            [14.0 / 3.0, 14.0 / 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 7.0],
        ]
    )

    np.testing.assert_allclose(da, expected_da, atol=1e-7)
    np.testing.assert_allclose(dase, expected_dase, atol=1e-6)
    maximal_storage_path(supply, dase.sum(axis=0), capacity)
    assert np.all(leontief_utilities(dase, demands) >= leontief_utilities(da, demands) - 1e-7)


if __name__ == "__main__":
    test_small_paper_example()
    test_four_by_four_finite_capacity_example()
    print("All DASE tests passed.")
