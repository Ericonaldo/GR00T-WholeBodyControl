"""
G1 Inspire Hand Controller

This module provides a unified interface for controlling the Unitree Inspire Hand
(used in G1 robot) through the Env base class interface, compatible with desktop control.

The Inspire Hand uses unitree_sdk2py DDS communication:
- Subscribe to "rt/inspire/state" for hand state (MotorStates_)
- Publish to "rt/inspire/cmd" for hand control (MotorCmds_)

Each hand has 6 DOF: [pinky, ring, middle, index, thumb_bend, thumb_rotation]
Position control range: [0, 1] where 0=close, 1=open
"""

import sys
import time

import gymnasium as gym
import numpy as np

from gr00t.control.base.env import Env
from gr00t.control.envs.g1.utils.command_sender import InspireHandCommandSender
from gr00t.control.envs.g1.utils.state_processor import InspireHandStateProcessor


class G1InspireHand(Env):
    """
    Unitree G1 Inspire Hand Controller

    Provides a unified interface for controlling the Inspire Hand through the Env base class.
    Supports both left and right hand control with 6 DOF each.

    Args:
        is_left (bool): If True, control left hand; if False, control right hand
    """

    # Inspire Hand has 6 DOF per hand
    NUM_DOF = 6

    # Joint names for reference
    JOINT_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"]

    def __init__(self, is_left: bool = True):
        super().__init__()
        self.is_left = is_left

        # Initialize state processor and command sender
        self.hand_state_processor = InspireHandStateProcessor(is_left=self.is_left)
        self.hand_command_sender = InspireHandCommandSender(is_left=self.is_left)

        # Hand offset for calibration (future use)
        self.hand_q_offset = np.zeros(self.NUM_DOF)

    def observe(self) -> dict[str, any]:
        """
        Observe current hand state

        Returns:
            dict with keys:
                - hand_q: Joint positions [6] (normalized 0-1, 0=close, 1=open)
                - hand_dq: Joint velocities [6]
                - hand_tau_est: Estimated joint torques [6]
        """
        hand_state = self.hand_state_processor._prepare_low_state()  # (1, 18)
        assert hand_state.shape == (1, 18), f"Expected shape (1, 18), got {hand_state.shape}"

        # Apply offset to the hand state
        hand_state[0, :6] = hand_state[0, :6] + self.hand_q_offset

        hand_q = hand_state[0, :6]
        hand_dq = hand_state[0, 6:12]
        hand_tau_est = hand_state[0, 12:18]

        return {
            "hand_q": hand_q,
            "hand_dq": hand_dq,
            "hand_tau_est": hand_tau_est,
        }

    def queue_action(self, action: dict[str, any]):
        """
        Send hand control command

        Args:
            action: dict with key:
                - hand_q: Target joint positions [6] (normalized 0-1, 0=close, 1=open)
        """
        if "hand_q" not in action:
            raise ValueError("action must contain 'hand_q' key")

        # Apply offset (for future calibration support)
        hand_q_target = action["hand_q"] - self.hand_q_offset

        # Send command
        self.hand_command_sender.send_command(hand_q_target)

    def observation_space(self) -> gym.Space:
        """Define observation space"""
        return gym.spaces.Dict(
            {
                "hand_q": gym.spaces.Box(low=0.0, high=1.0, shape=(self.NUM_DOF,), dtype=np.float32),
                "hand_dq": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.NUM_DOF,), dtype=np.float32
                ),
                "hand_tau_est": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.NUM_DOF,), dtype=np.float32
                ),
            }
        )

    def action_space(self) -> gym.Space:
        """Define action space"""
        return gym.spaces.Dict(
            {"hand_q": gym.spaces.Box(low=0.0, high=1.0, shape=(self.NUM_DOF,), dtype=np.float32)}
        )

    def set_predefined_pose(self, pose_name: str):
        """
        Set hand to a predefined pose

        Args:
            pose_name: One of "open", "close", "half"
        """
        predefined_poses = {
            "open": np.ones(self.NUM_DOF, dtype=np.float32),
            "close": np.zeros(self.NUM_DOF, dtype=np.float32),
            "half": np.full(self.NUM_DOF, 0.5, dtype=np.float32),
        }

        if pose_name not in predefined_poses:
            raise ValueError(
                f"Invalid pose_name: {pose_name}. Must be one of {list(predefined_poses.keys())}"
            )

        self.queue_action({"hand_q": predefined_poses[pose_name]})

    def calibrate_hand(self):
        """
        Calibrate hand (placeholder for future implementation)

        Currently, the Inspire Hand uses normalized 0-1 control,
        so calibration may not be necessary. This method is kept
        for interface compatibility with G1ThreeFingerHand.
        """
        print("Note: Inspire Hand uses normalized 0-1 control.")
        print("Calibration not required. Setting offset to zero.")
        self.hand_q_offset = np.zeros(self.NUM_DOF)


def main():
    """
    Main function for testing G1InspireHand

    Tests basic hand control functionality with predefined poses and smooth interpolation.
    """
    print("=" * 60)
    print("G1 Inspire Hand Control Test")
    print("=" * 60)

    # Parse network interface from command line (optional)
    network_interface = sys.argv[1] if len(sys.argv) > 1 else None

    # Initialize DDS
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        if network_interface:
            print(f"Initializing DDS with network interface: {network_interface}")
            ChannelFactoryInitialize(0, network_interface)
        else:
            print("Initializing DDS with auto-detected network interface")
            ChannelFactoryInitialize(0)
    except Exception as e:
        print(f"Error initializing DDS: {e}")
        return 1

    # Create right hand controller
    print("\nCreating right hand controller...")
    hand = G1InspireHand(is_left=False)

    try:
        # Test 1: Predefined poses
        print("\n" + "=" * 60)
        print("Test 1: Predefined Poses")
        print("=" * 60)

        for pose in ["open", "close", "half"]:
            print(f"\nMoving to '{pose}' position...")
            hand.set_predefined_pose(pose)
            time.sleep(2.0)

            # Observe and display hand state
            obs = hand.observe()
            print(f"Hand position: {obs['hand_q']}")
            print(f"Hand velocity: {obs['hand_dq']}")
            print(f"Finger states:")
            for i, name in enumerate(G1InspireHand.JOINT_NAMES):
                state = "OPEN" if obs["hand_q"][i] > 0.5 else "CLOSE"
                print(f"  {name:15s}: {obs['hand_q'][i]:.3f} ({state})")

        # Test 2: Custom position
        print("\n" + "=" * 60)
        print("Test 2: Custom Position")
        print("=" * 60)
        print("\nSetting custom position:")
        print("  Pinky & ring: open (1.0)")
        print("  Middle & index: half (0.5)")
        print("  Thumb: closed (0.0)")

        custom_pose = np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        hand.queue_action({"hand_q": custom_pose})
        time.sleep(2.0)

        obs = hand.observe()
        print(f"\nActual position: {obs['hand_q']}")

        # Test 3: Smooth interpolation
        print("\n" + "=" * 60)
        print("Test 3: Smooth Interpolation (open -> close)")
        print("=" * 60)
        print("\nInterpolating over 2.5 seconds...")

        steps = 50
        for i in range(steps + 1):
            alpha = i / steps
            # Interpolate from open (1.0) to close (0.0)
            target = np.ones(6, dtype=np.float32) * (1.0 - alpha)
            hand.queue_action({"hand_q": target})
            time.sleep(0.05)  # 50ms between steps

        print("Interpolation complete!")

        # Test 4: Individual finger control
        print("\n" + "=" * 60)
        print("Test 4: Individual Finger Control")
        print("=" * 60)

        # Start with open hand
        current_pose = np.ones(6, dtype=np.float32)

        for i, finger_name in enumerate(G1InspireHand.JOINT_NAMES):
            print(f"\nClosing {finger_name}...")
            current_pose[i] = 0.0
            hand.queue_action({"hand_q": current_pose.copy()})
            time.sleep(1.0)

            obs = hand.observe()
            print(f"Position: {obs['hand_q']}")

        # Test 5: Continuous monitoring
        print("\n" + "=" * 60)
        print("Test 5: Continuous Monitoring (10 iterations)")
        print("=" * 60)

        # Set to half-open
        hand.set_predefined_pose("half")

        print("\nMonitoring hand state (press Ctrl+C to stop early)...")
        for iteration in range(10):
            obs = hand.observe()
            print(f"\nIteration {iteration + 1}/10:")
            print(f"  Position: {obs['hand_q']}")
            print(f"  Velocity: {obs['hand_dq']}")
            print(f"  Torque:   {obs['hand_tau_est']}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError during test: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Return to open position
        print("\n" + "=" * 60)
        print("Cleanup: Returning to open position...")
        print("=" * 60)
        hand.set_predefined_pose("open")
        time.sleep(1.0)
        print("Test complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
