from dataclasses import dataclass
from oren.utils.config_abc import ConfigABC
from rclpy.duration import Duration
from rclpy.qos import (
    QoSProfile,
    QoSPresetProfiles,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSLivelinessPolicy,
)


@dataclass
class RosTopicParam(ConfigABC):
    path: str = ""

    # "system_default", "services_default", "sensor_data", "parameter_events", "parameter_updates", "action_status_default"
    qos_preset: str = ""

    qos_history: str = "keep_last"
    qos_depth: int = 10
    qos_reliability: str = "reliable"
    qos_durability: str = "volatile"
    qos_deadline_sec: int = 0
    qos_deadline_nanosec: int = 0
    qos_lifespan_sec: int = 0
    qos_lifespan_nanosec: int = 0
    qos_liveliness: str = "automatic"
    qos_liveliness_lease_duration_sec: int = 0
    qos_liveliness_lease_duration_nanosec: int = 0
    avoid_ros_namespace_conventions: bool = False

    def declare_ros_params(self, ros_node, prefix: str = ""):
        if prefix:
            prefix = prefix.rstrip(".") + "."

        declare_fn = lambda name, default: ros_node.declare_parameter(prefix + name, default)

        declare_fn("path", self.path)
        declare_fn("qos_preset", self.qos_preset)
        declare_fn("qos_history", self.qos_history)
        declare_fn("qos_depth", self.qos_depth)
        declare_fn("qos_reliability", self.qos_reliability)
        declare_fn("qos_durability", self.qos_durability)
        declare_fn("qos_deadline_sec", self.qos_deadline_sec)
        declare_fn("qos_deadline_nanosec", self.qos_deadline_nanosec)
        declare_fn("qos_lifespan_sec", self.qos_lifespan_sec)
        declare_fn("qos_lifespan_nanosec", self.qos_lifespan_nanosec)
        declare_fn("qos_liveliness", self.qos_liveliness)
        declare_fn("qos_liveliness_lease_duration_sec", self.qos_liveliness_lease_duration_sec)
        declare_fn("qos_liveliness_lease_duration_nanosec", self.qos_liveliness_lease_duration_nanosec)
        declare_fn("avoid_ros_namespace_conventions", self.avoid_ros_namespace_conventions)

    def update_from_ros_params(self, ros_node, prefix: str = ""):
        if prefix:
            prefix = prefix.rstrip(".") + "."

        get_fn = lambda name: ros_node.get_parameter(prefix + name).value

        self.path = get_fn("path")
        self.qos_preset = get_fn("qos_preset")
        self.qos_history = get_fn("qos_history")
        self.qos_depth = int(get_fn("qos_depth"))
        self.qos_reliability = get_fn("qos_reliability")
        self.qos_durability = get_fn("qos_durability")
        self.qos_deadline_sec = int(get_fn("qos_deadline_sec"))
        self.qos_deadline_nanosec = int(get_fn("qos_deadline_nanosec"))
        self.qos_lifespan_sec = int(get_fn("qos_lifespan_sec"))
        self.qos_lifespan_nanosec = int(get_fn("qos_lifespan_nanosec"))
        self.qos_liveliness = get_fn("qos_liveliness")
        self.qos_liveliness_lease_duration_sec = int(get_fn("qos_liveliness_lease_duration_sec"))
        self.qos_liveliness_lease_duration_nanosec = int(get_fn("qos_liveliness_lease_duration_nanosec"))
        self.avoid_ros_namespace_conventions = bool(get_fn("avoid_ros_namespace_conventions"))

    def get_qos(self):
        if self.qos_preset:
            # Rebuild from the preset's fields rather than mutating the singleton
            # `QoSPresetProfiles.<NAME>.value` (which would bleed into every other
            # subscriber using the same preset). deepcopy can't be used because the
            # underlying rclpy duration objects aren't picklable.
            preset = getattr(QoSPresetProfiles, self.qos_preset.upper()).value
            qos = QoSProfile(
                history=preset.history,
                depth=preset.depth,
                reliability=preset.reliability,
                durability=preset.durability,
                deadline=preset.deadline,
                lifespan=preset.lifespan,
                liveliness=preset.liveliness,
                liveliness_lease_duration=preset.liveliness_lease_duration,
                avoid_ros_namespace_conventions=preset.avoid_ros_namespace_conventions,
            )
        else:
            # QoSProfile() requires history+depth at construction time; seed with
            # the configured values, the block below will overwrite as needed.
            qos = QoSProfile(
                history=getattr(QoSHistoryPolicy, (self.qos_history or "keep_last").upper()),
                depth=self.qos_depth if self.qos_depth > 0 else 10,
            )

        if self.qos_history:
            qos.history = getattr(QoSHistoryPolicy, self.qos_history.upper())
        if self.qos_depth > 0:
            qos.depth = self.qos_depth
        if self.qos_reliability:
            qos.reliability = getattr(QoSReliabilityPolicy, self.qos_reliability.upper())
        if self.qos_durability:
            qos.durability = getattr(QoSDurabilityPolicy, self.qos_durability.upper())
        if self.qos_deadline_sec > 0 or self.qos_deadline_nanosec > 0:
            qos.deadline = Duration(seconds=self.qos_deadline_sec, nanoseconds=self.qos_deadline_nanosec)
        if self.qos_lifespan_sec > 0 or self.qos_lifespan_nanosec > 0:
            qos.lifespan = Duration(seconds=self.qos_lifespan_sec, nanoseconds=self.qos_lifespan_nanosec)
        if self.qos_liveliness:
            qos.liveliness = getattr(QoSLivelinessPolicy, self.qos_liveliness.upper())
        if self.qos_liveliness_lease_duration_sec > 0 or self.qos_liveliness_lease_duration_nanosec > 0:
            qos.liveliness_lease_duration = Duration(
                seconds=self.qos_liveliness_lease_duration_sec,
                nanoseconds=self.qos_liveliness_lease_duration_nanosec,
            )
        qos.avoid_ros_namespace_conventions = self.avoid_ros_namespace_conventions
        return qos
