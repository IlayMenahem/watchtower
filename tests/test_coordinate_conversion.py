import torch
import math
from watchtower.delta_estimator import cartesian_to_spherical, spherical_to_cartesian


def test_1d_coordinates():
    """Test 1D coordinate conversion."""
    test_cases = [
        [5.0],
        [-3.0],
        [0.0],
        [1e-8],
        [1e8],
    ]

    for point in test_cases:
        x = torch.tensor(point)
        spherical = cartesian_to_spherical(x)
        cartesian = spherical_to_cartesian(spherical)

        assert torch.allclose(x, cartesian, atol=1e-6), f"1D conversion failed for {point}"
        assert torch.allclose(spherical, x, atol=1e-6), f"1D spherical conversion failed for {point}"



def test_2d_coordinates():
    """Test 2D coordinate conversion."""
    test_cases = [
        [1.0, 0.0],    # Point on x-axis
        [0.0, 1.0],    # Point on y-axis
        [1.0, 1.0],    # 45 degree point
        [-1.0, 0.0],   # Negative x-axis
        [0.0, -1.0],   # Negative y-axis
        [3.0, 4.0],    # 3-4-5 triangle
        [-2.0, -2.0],  # Third quadrant
        [5.0, -12.0],  # Fourth quadrant
        [0.0, 0.0],    # Origin
    ]

    for point in test_cases:
        x = torch.tensor(point)
        spherical = cartesian_to_spherical(x)
        cartesian = spherical_to_cartesian(spherical)
        assert torch.allclose(x, cartesian, atol=1e-6), f"2D conversion failed for {point}"


def test_3d_coordinates():
    """Test 3D coordinate conversion."""
    test_cases = [
        [1.0, 0.0, 0.0],    # x-axis
        [0.0, 1.0, 0.0],    # y-axis
        [0.0, 0.0, 1.0],    # z-axis
        [1.0, 1.0, 1.0],    # Equal coordinates
        [3.0, 4.0, 0.0],    # xy-plane
        [0.0, 3.0, 4.0],    # yz-plane
        [4.0, 0.0, 3.0],    # xz-plane
        [1.0, 2.0, 3.0],    # General point
        [-1.0, -2.0, -3.0], # Negative octant
        [0.0, 0.0, 0.0],    # Origin
    ]

    for point in test_cases:
        x = torch.tensor(point)
        spherical = cartesian_to_spherical(x)
        cartesian = spherical_to_cartesian(spherical)
        assert torch.allclose(x, cartesian, atol=1e-6), f"3D conversion failed for {point}"


def test_4d_coordinates():
    """Test 4D coordinate conversion."""
    test_cases = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0, 5.0],
        [-1.0, -2.0, 3.0, -4.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    for point in test_cases:
        x = torch.tensor(point)
        spherical = cartesian_to_spherical(x)
        cartesian = spherical_to_cartesian(spherical)
        assert torch.allclose(x, cartesian, atol=1e-6), f"4D conversion failed for {point}"


def test_higher_dimensions():
    """Test higher dimensional coordinate conversion."""
    # Test 5D
    x5d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    spherical5d = cartesian_to_spherical(x5d)
    cartesian5d = spherical_to_cartesian(spherical5d)
    assert torch.allclose(x5d, cartesian5d, atol=1e-6), "5D conversion failed"

    # Test 6D
    x6d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    spherical6d = cartesian_to_spherical(x6d)
    cartesian6d = spherical_to_cartesian(spherical6d)
    assert torch.allclose(x6d, cartesian6d, atol=1e-6), "6D conversion failed"

    # Test 10D
    x10d = torch.randn(10)
    spherical10d = cartesian_to_spherical(x10d)
    cartesian10d = spherical_to_cartesian(spherical10d)
    assert torch.allclose(x10d, cartesian10d, atol=1e-5), "10D conversion failed"


def test_edge_cases():
    """Test edge cases."""
    # Empty tensor
    x_empty = torch.tensor([])
    spherical_empty = cartesian_to_spherical(x_empty)
    cartesian_empty = spherical_to_cartesian(spherical_empty)
    assert len(spherical_empty) == 0 and len(cartesian_empty) == 0, "Empty tensor test failed"

    # Very small values
    x_small = torch.tensor([1e-12, 1e-12])
    spherical_small = cartesian_to_spherical(x_small)
    cartesian_small = spherical_to_cartesian(spherical_small)
    assert torch.allclose(x_small, cartesian_small, atol=1e-10), "Small values test failed"

    # Large values
    x_large = torch.tensor([1e6, 1e6, 1e6])
    spherical_large = cartesian_to_spherical(x_large)
    cartesian_large = spherical_to_cartesian(spherical_large)
    assert torch.allclose(x_large, cartesian_large, atol=1e-3), "Large values test failed"


def test_radius_preservation():
    """Test that radius is correctly preserved."""
    for dim in [2, 3, 4, 5, 6]:
        x = torch.randn(dim)
        spherical = cartesian_to_spherical(x)
        expected_radius = torch.sqrt(torch.sum(x**2))
        assert torch.allclose(spherical[0], expected_radius, rtol=1e-6), f"Radius not preserved in {dim}D"


def test_specific_angles():
    """Test specific angle cases for 2D."""
    # Test known angles
    test_cases = [
        ([1.0, 0.0], 0.0),              # 0 degrees
        ([0.0, 1.0], math.pi/2),        # 90 degrees
        ([-1.0, 0.0], math.pi),         # 180 degrees
        ([0.0, -1.0], -math.pi/2),      # -90 degrees
        ([1.0, 1.0], math.pi/4),        # 45 degrees
        ([-1.0, -1.0], -3*math.pi/4),   # -135 degrees
    ]

    for point, expected_angle in test_cases:
        x = torch.tensor(point)
        spherical = cartesian_to_spherical(x)
        # Check angle (second component)
        if len(spherical) > 1:
            actual_angle = spherical[1].item()
            assert abs(actual_angle - expected_angle) < 1e-6, f"Angle mismatch for {point}: expected {expected_angle}, got {actual_angle}"


def test_zero_handling():
    """Test handling of zero coordinates."""
    # Origin in various dimensions
    for dim in [1, 2, 3, 4, 5]:
        x_zero = torch.zeros(dim)
        spherical_zero = cartesian_to_spherical(x_zero)
        cartesian_zero = spherical_to_cartesian(spherical_zero)
        assert torch.allclose(x_zero, cartesian_zero, atol=1e-10), f"Zero handling failed in {dim}D"
        assert spherical_zero[0] == 0.0, f"Radius should be 0 for origin in {dim}D"


def test_random_points():
    """Test with random points to ensure robustness."""
    torch.manual_seed(42)  # For reproducibility

    for dim in [2, 3, 4, 5]:
        for _ in range(20):  # Test 20 random points per dimension
            x = torch.randn(dim) * 10  # Scale up for variety
            spherical = cartesian_to_spherical(x)
            cartesian = spherical_to_cartesian(spherical)
            assert torch.allclose(x, cartesian, atol=1e-5), f"Random point conversion failed in {dim}D"


def test_dtype_preservation():
    """Test that data types are preserved."""
    x_float32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    spherical = cartesian_to_spherical(x_float32)
    cartesian = spherical_to_cartesian(spherical)

    assert spherical.dtype == torch.float32, "Spherical dtype not preserved"
    assert cartesian.dtype == torch.float32, "Cartesian dtype not preserved"


def test_device_preservation():
    """Test that device is preserved."""
    x = torch.tensor([1.0, 2.0, 3.0])
    spherical = cartesian_to_spherical(x)
    cartesian = spherical_to_cartesian(spherical)

    assert spherical.device == x.device, "Spherical device not preserved"
    assert cartesian.device == x.device, "Cartesian device not preserved"


def benchmark_performance():
    """Benchmark the performance of coordinate conversion."""
    import time

    print("\n=== Performance Benchmark ===")

    dimensions = [2, 3, 4, 5, 10, 20]
    num_iterations = 1000

    for dim in dimensions:
        x = torch.randn(dim)

        # Benchmark cartesian_to_spherical
        start_time = time.time()
        for _ in range(num_iterations):
            spherical = cartesian_to_spherical(x)
        cart_to_sph_time = time.time() - start_time

        # Benchmark spherical_to_cartesian
        start_time = time.time()
        for _ in range(num_iterations):
            cartesian = spherical_to_cartesian(spherical)
        sph_to_cart_time = time.time() - start_time

        print(f"{dim}D - Cart->Sph: {cart_to_sph_time:.4f}s, Sph->Cart: {sph_to_cart_time:.4f}s")


def run_comprehensive_tests():
    """Run all tests and provide a summary."""
    print("=== Testing n-Dimensional Coordinate Conversion ===")

    test_functions = [
        ("1D Coordinates", test_1d_coordinates),
        ("2D Coordinates", test_2d_coordinates),
        ("3D Coordinates", test_3d_coordinates),
        ("4D Coordinates", test_4d_coordinates),
        ("Higher Dimensions", test_higher_dimensions),
        ("Edge Cases", test_edge_cases),
        ("Radius Preservation", test_radius_preservation),
        ("Specific Angles", test_specific_angles),
        ("Zero Handling", test_zero_handling),
        ("Random Points", test_random_points),
        ("Data Type Preservation", test_dtype_preservation),
        ("Device Preservation", test_device_preservation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {str(e)}")
            failed += 1

    print(f"\nTest Summary: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        benchmark_performance()
    else:
        print("❌ Some tests failed.")

    return failed == 0


if __name__ == "__main__":
    run_comprehensive_tests()
