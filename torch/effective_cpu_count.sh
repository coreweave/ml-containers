#!/bin/sh

CPU_QUOTA() (
    CGROUP='/sys/fs/cgroup';
    CGROUP_V1="$CGROUP/cpu,cpuacct";
    CGROUP_V1_QUOTA="$CGROUP_V1/cpu.cfs_quota_us";
    CGROUP_V1_PERIOD="$CGROUP_V1/cpu.cfs_period_us";
    CGROUP_V2="$CGROUP/user.slice/cpu.max";
    if [ ! -d "$CGROUP" ]; then
        return 1;
    elif [ -f "$CGROUP_V1_QUOTA" ] && [ -f "$CGROUP_V1_PERIOD" ]; then
        IFS='' read -r QUOTA 2> /dev/null < "$CGROUP_V1_QUOTA" || return 1;
        IFS='' read -r PERIOD 2> /dev/null < "$CGROUP_V1_PERIOD" || return 1;
    elif [ -f "$CGROUP_V2" ]; then
        IFS=' ' read -r QUOTA PERIOD 2> /dev/null < "$CGROUP_V2" || return 1;
    else
        return 1;
    fi;

    if [ "$QUOTA" -gt 0 ] 2> /dev/null && [ "$PERIOD" -gt 0 ] 2> /dev/null; then
        echo $((QUOTA / PERIOD));
        return 0;
    else
        return 1;
    fi;
)

EFFECTIVE_CPU_COUNT() {
    CPU_QUOTA || getconf _NPROCESSORS_ONLN;
}

EFFECTIVE_CPU_COUNT;
