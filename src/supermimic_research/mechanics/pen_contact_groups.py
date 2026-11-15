"""Fn6 is fingers1..5 plus other robot, not six selected collision geoms."""


def normal_group(name):
    if not name.startswith('robot/'):return -1
    return next((i-1 for i in range(1,6) if name.startswith(f'robot/right_finger{i}_')),5)
