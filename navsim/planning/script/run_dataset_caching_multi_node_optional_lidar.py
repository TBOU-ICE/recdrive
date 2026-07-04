"""Dataset caching entrypoint with optional lidar-path support.

This wrapper leaves the stock caching implementation unchanged. It only applies
an explicit runtime patch before delegating to ``run_dataset_caching_multi_node``.
"""

from navsim.common.optional_lidar_patch import apply_optional_lidar_patch

apply_optional_lidar_patch()

from navsim.planning.script.run_dataset_caching_multi_node import main


if __name__ == "__main__":
    main()
