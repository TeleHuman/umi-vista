#!/bin/bash

sudo sh -c 'echo 256 > /sys/module/usbcore/parameters/usbfs_memory_mb'

source ~/galaxea/install/setup.bash
cd ~/galaxea/install/startup_config/share/startup_config/script
./robot_startup.sh boot ../sessions.d/ATCStandard/R1PROBody.d/

# Give the system a moment to settle after startup.
sleep 1

# Find and kill processes using /dev/video0..7.
# Uses fuser if available (preferred), otherwise falls back to lsof.
kill_video_users() {
	local dev
	for dev in /dev/video{0..7}; do
		[[ -e "$dev" ]] || continue

		local pids=""
		if command -v fuser >/dev/null 2>&1; then
			# fuser output is space-separated PIDs on stdout.
			pids="$(sudo fuser -f "$dev" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ' | xargs echo -n 2>/dev/null)"
		elif command -v lsof >/dev/null 2>&1; then
			pids="$(sudo lsof -t "$dev" 2>/dev/null | sort -u | tr '\n' ' ' | xargs echo -n 2>/dev/null)"
		else
			echo "No fuser/lsof found; cannot detect users of $dev"
			continue
		fi

		if [[ -z "$pids" ]]; then
			continue
		fi

		echo "Killing processes using $dev: $pids"
		sudo kill -TERM $pids 2>/dev/null || true
		sleep 0.5
		# Force-kill any that remain.
		if command -v fuser >/dev/null 2>&1; then
			local still="$(sudo fuser -f "$dev" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ' | xargs echo -n 2>/dev/null)"
			[[ -z "$still" ]] || sudo kill -KILL $still 2>/dev/null || true
		elif command -v lsof >/dev/null 2>&1; then
			local still="$(sudo lsof -t "$dev" 2>/dev/null | sort -u | tr '\n' ' ' | xargs echo -n 2>/dev/null)"
			[[ -z "$still" ]] || sudo kill -KILL $still 2>/dev/null || true
		fi
	done
}

kill_video_users
#source /opt/ros/humble/setup.bash
#source ~/galaxea/install/setup.bash

export ROBOT_NAME="r1pro"
