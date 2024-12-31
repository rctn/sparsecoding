import torch

from sparsecoding.transforms import patchify, quilt, sample_random_patches


def test_patchify_quilt_cycle():
    X, Y, Z = 3, 4, 5
    C = 3
    P = 8
    H = 6 * P
    W = 8 * P

    images = torch.rand((X, Y, Z, C, H, W), dtype=torch.float32)

    patches = patchify(P, images)
    assert patches.shape == (X, Y, Z, int(H / P) * int(W / P), C, P, P)

    quilted_images = quilt(H, W, patches)
    assert torch.allclose(
        images,
        quilted_images,
    ), "Quilted images should be equal to input images."


def test_sample_random_patches():
    X, Y, Z = 3, 4, 5
    C = 3
    P = 8
    H = 4 * P
    W = 8 * P
    N = 10

    images = torch.rand((X, Y, Z, C, H, W), dtype=torch.float32)

    random_patches = sample_random_patches(P, N, images)
    assert random_patches.shape == (N, C, P, P)

    # Check that patches are actually taken from one of the images.
    all_patches = torch.nn.functional.unfold(
        input=images.reshape(-1, C, H, W),
        kernel_size=P,
    )  # [prod(*), C*P*P, L]
    all_patches = torch.permute(all_patches, (0, 2, 1))  # [prod(*), L, C*P*P]
    all_patches = torch.reshape(all_patches, (-1, C * P * P))
    for n in range(N):
        patch = random_patches[n].reshape(1, C * P * P)
        delta = torch.abs(patch - all_patches)  # [-1, C*P*P]
        patchwise_delta = torch.sum(delta, dim=1)  # [-1]
        assert torch.min(patchwise_delta) == 0.0
