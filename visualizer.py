import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import viz_cfg
from files import create_path

class FuelDump:
    def __init__(self, x, volume, withdraw_size):
        self.x = x
        self.volume = volume
        self.withdraw_size = withdraw_size

        print("Created fuel dump at x:", x, "with volume:", volume, "and withdraw size:", withdraw_size)


class Visualizer:
    def __init__(self, save_path=viz_cfg.save_path):
        self.save_path = save_path

        create_path(save_path)

    def get_depo(self, k, n):
        factor = 1 / (2 * n - 2 * k + 1)
        new_x = factor + self.fuel_dumps[-1].x
        new_volume = (2 * n - 2 * k - 1) * factor
        new_withdraw_size = factor

        new_dump = FuelDump(new_x, new_volume, new_withdraw_size)

        return new_dump

    def simulate_trip(self, n=viz_cfg.n):
        self.fuel_dumps = [FuelDump(0, 0, 0)]

        for k in range(n - 1):
            new_depo = self.get_depo(k, n - 1)
            self.fuel_dumps.append(new_depo)

        total_dist = 1 + sum([fuel.withdraw_size for fuel in self.fuel_dumps])
        self.fuel_dumps.append(FuelDump(total_dist, 0, 0))

    def generate_trips(self, dists, slow_factor, reverse=False):
        trips = []

        iterand = [element[::-1] for element in dists[::-1]] if reverse else dists
        for start, end in iterand:
            dist = abs(end - start)
            n_frames = int(dist * slow_factor)

            trips.extend(np.linspace(start, end, n_frames))

        return trips

    def visualize_trip(self, slow_factor=viz_cfg.slow_factor, fps=viz_cfg.fps, paus_time=viz_cfg.paus_time):
        xs = []
        fuels = []
        trips = []
        fuel_changes = []

        for fuel_dump in self.fuel_dumps[1:]:
            fuel_x = np.array([fuel.x for fuel in self.fuel_dumps[1:]])

            passed = np.array(self.fuel_dumps[1:])[fuel_dump.x > fuel_x]
            x = []

            dists = [(passed[i - 1].x if i > 0 else 0, passed_fuel.x) for i, passed_fuel in enumerate(passed)] + [(passed[-1].x if len(passed) else 0, fuel_dump.x)]
            forward_trips = self.generate_trips(dists, slow_factor)
            x.extend(forward_trips)

            if fuel_dump != self.fuel_dumps[-1]:
                backward_trips = self.generate_trips(dists, slow_factor, reverse=True)  # forward_trips[::-1]
                x.extend(backward_trips)

                x = np.array(x)
                deposited_fuel = [0] * len(forward_trips) + [fuel_dump.volume] * len(forward_trips)
                a = -np.diff(backward_trips, prepend=backward_trips[0])
                h = np.cumsum(a)
                x_covered = np.concatenate((forward_trips, h + forward_trips[-1]))
            else:
                x = np.array(x)
                deposited_fuel = [0] * len(x)
                x_covered = x

            start_point = len(xs)  # trips[-1][2] if len(trips) else 0
            mid_point = start_point + int(len(x) // 2)
            end_point = start_point + len(x)
            trips.append([start_point, mid_point, end_point])

            xs.extend(x)

            fuel_x = np.array([fuel.x for fuel in self.fuel_dumps if fuel != fuel_dump])
            withdraw_volumes = np.array([fuel.withdraw_size for fuel in self.fuel_dumps if fuel != fuel_dump])

            is_stopped = np.tile(np.diff(x, prepend=0)[:, None] == 0, (1, len(fuel_x)))
            pos_passed_fuel_mask11 = (x[:, None] >= fuel_x[None]) & is_stopped
            pos_passed_fuel_mask = np.maximum.accumulate(pos_passed_fuel_mask11, axis=0)

            if fuel_dump != self.fuel_dumps[-1]:
                furthest_out = np.maximum.accumulate(x[:, None])
                neg_passed_fuel_mask11 = (x[:, None] <= fuel_x[None]) & is_stopped & (furthest_out > fuel_x[None])
                neg_passed_fuel_mask = np.maximum.accumulate(neg_passed_fuel_mask11, axis=0)
            else:
                neg_passed_fuel_mask = np.zeros_like(pos_passed_fuel_mask)

            withdrawn_fuels_multiplier = np.array(pos_passed_fuel_mask, dtype=int) + np.array(neg_passed_fuel_mask, dtype=int)
            withdrawn_fuels_volume = withdrawn_fuels_multiplier * withdraw_volumes
            total_withdrawn_fuels_volume = np.sum(withdrawn_fuels_volume, axis=-1)
            # c = b[a]

            net_fuel = total_withdrawn_fuels_volume - np.array(deposited_fuel)

            fuel = 1 - x_covered + net_fuel
            fuels.extend(fuel)

            net_fuel_changes = np.diff(net_fuel, axis=0)
            net_fuel_change_indices = np.nonzero(net_fuel_changes)

            fuel_changes.extend(zip(start_point + net_fuel_change_indices[0] + 1, net_fuel_changes[net_fuel_change_indices]))

        self.plot_visulization(np.array(xs), np.array(fuels), np.array(trips), fuel_changes, fps=fps, paus_time=paus_time)

    def plot_visulization(self, xs, fuels, trips, fuel_changes, fps, paus_time):
        fig, axs = plt.subplots(len(trips) + 1, figsize=(7, 2 + 1.5 * len(trips)), sharex=True)
        axs[0].set_xlim(0, np.max(xs))
        height = 1.5
        axs[0].set_ylim(0, height)
        axs[0].set_yticks([])

        point, = axs[0].plot([], [], "bo", markersize=8, label="Jeep")
        text = axs[0].text(0, 0, "", ha="center", va="bottom", fontsize=10, color="orange")
        text2 = axs[0].text(0, 0, "", ha="center", va="bottom", fontsize=10)

        lines = []
        for i, trip in enumerate(trips):
            line = axs[i + 1].plot([], [], color="orange")
            lines.append(line[0])

            axs[i + 1].set_xlim(0, np.max(xs))
            axs[i + 1].set_ylim(0, 1)
            axs[i + 1].set_title(f"Trip {i + 1}")
            axs[i + 1].set_ylabel("Fuel Level")

        plt.xlabel("Displacement")

        """  # Only stops at midpoints of trips
        paus_frames = int(fps * paus_time)
        index_script = []
        for i, trip in enumerate(trips[:, 1]):
            prev_trip = trips[i - 1][1] if i > 0 else 0
            trip_script = list(range(prev_trip, trip)) + [trip - 1] * paus_frames

            index_script += trip_script
        """

        # """  # Stops at every fuel change
        paus_frames = int(fps * paus_time)
        index_script = []
        for i, fuel_change in enumerate(fuel_changes):
            prev_fuel_change = fuel_changes[i - 1][0] if i > 0 else 0
            index_script += list(range(prev_fuel_change, fuel_change[0] + 1)) + [fuel_change[0]] * paus_frames

        index_script += list(range(fuel_changes[-1][0], len(xs))) + [len(xs) - 1] * paus_frames
        # """

        def animate(i):
            index = index_script[i]

            all_trips = np.array(trips)
            trip_index = int(np.nonzero((index >= all_trips[:, 0]) & (index < all_trips[:, 2]))[0][0])
            start_index = int(all_trips[trip_index][0])
            lines[trip_index].set_data(xs[start_index:index + 1], fuels[start_index:index + 1])

            if trip_index == 4:
                pass

            # print("Animating frame", i, "index:", index)
            x = xs[index]
            point.set_data([x], [height / 2])
            text.set_position((x, height / 2 + 0.1))

            np_fuel_change = np.array(fuel_changes)
            if index in np_fuel_change[:, 0]:
                fuel_change = np.sum(np_fuel_change[np_fuel_change[:, 0] == index][:, 1])
                # print("Fuel change at index", index, ":", fuel_change)
                text.set_text(f"Fuel: {(fuels[index] - fuel_change):.2f}")

                text2.set_position((x, height / 2 + 0.3))
                change_text = ("+" if fuel_change > 0 else "-") + f"{abs(fuel_change):.2f}"
                text2.set_text(change_text)
                text2.set_color("green" if fuel_change >= 0 else "red")
            else:
                text.set_text(f"Fuel: {fuels[index]:.2f}")
                text2.set_text("")

            return point, text

        plt.tight_layout()
        # plt.legend()

        ani = animation.FuncAnimation(fig, animate, frames=len(index_script))
        animation_path = os.path.join(self.save_path, "animation.gif")
        ani.save(animation_path, fps=fps)


if __name__ == "__main__":
    a = Visualizer()
    a.simulate_trip(10)
    a.visualize_trip()
