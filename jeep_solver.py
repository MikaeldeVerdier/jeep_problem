import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import jeep_solver_config
from files import create_path

class JeepSolver:
    def __init__(self, save_path=jeep_solver_config.save_path):
        self.save_path = save_path

        create_path(save_path)
        self.images_path = os.path.join(save_path, "images")
        create_path(self.images_path)
        self.animations_path = os.path.join(save_path, "animations")
        create_path(self.animations_path)

    def get_depo(self, k, n):
        factor = 1 / (2 * n - 2 * k + 1)
        new_x = factor + (self.fuel_dumps[-1][0] if len(self.fuel_dumps) else 0)
        new_volume = 1 - 2 * factor  # (2 * n - 2 * k - 1) * factor
        new_withdraw_size = factor

        new_dump = [new_x, new_volume, new_withdraw_size]

        print("Depo created at", new_dump[0], "with volume", new_dump[1], "and withdraw size", new_dump[2])

        return new_dump

    def simulate_trip(self, n=jeep_solver_config.n):
        self.fuel_dumps = []

        for k in range(1, n):  # could do range(n - 1) and self.get_depo(k, n - 1) instead to use 0-indices instead
            new_depo = self.get_depo(k, n)
            self.fuel_dumps.append(new_depo)

        self.fuel_dumps = np.array(self.fuel_dumps).reshape(-1, 3)
        total_dist = 1 + np.sum(self.fuel_dumps[:, 2])
        self.fuel_dumps = np.vstack((self.fuel_dumps, [total_dist, 0, 0]))

        print(f"Final distance crossed for n={n} is {total_dist}")

    def generate_trips(self, dists, slow_factor, reverse=False):
        trips = []

        iterand = [element[::-1] for element in dists[::-1]] if reverse else dists
        for start, end in iterand:
            dist = abs(end - start)
            n_frames = int(dist * slow_factor)

            trips.extend(np.linspace(start, end, n_frames))

        return trips

    def visualize_trip(self, slow_factor=jeep_solver_config.slow_factor, fps=jeep_solver_config.fps, paus_time=jeep_solver_config.paus_time, do_animation=jeep_solver_config.do_animation):
        xs = []
        fuels = []
        trips = []
        fuel_changes = []

        for i, fuel_dump in enumerate(self.fuel_dumps):
            is_last_iteration = i == len(self.fuel_dumps) - 1

            passed = self.fuel_dumps[fuel_dump[0] > self.fuel_dumps[:, 0]]
            x = []

            dists = [(passed[i - 1][0] if i > 0 else 0, passed_fuel[0]) for i, passed_fuel in enumerate(passed)] + [(passed[-1][0] if len(passed) else 0, fuel_dump[0])]
            forward_trips = self.generate_trips(dists, slow_factor)
            x.extend(forward_trips)

            if not is_last_iteration:
                backward_trips = self.generate_trips(dists, slow_factor, reverse=True)  # forward_trips[::-1]
                x.extend(backward_trips)

                x = np.array(x)
                deposited_fuel = [0] * len(forward_trips) + [fuel_dump[1]] * len(backward_trips)
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

            fuel_x = np.delete(self.fuel_dumps, i, axis=0)[:, 0]
            withdraw_volumes = np.delete(self.fuel_dumps, i, axis=0)[:, 2]

            is_stopped = np.tile(np.diff(x, prepend=0)[:, None] == 0, (1, len(fuel_x)))
            pos_passed_fuel_mask_full = (x[:, None] >= fuel_x[None]) & is_stopped
            pos_passed_fuel_mask = np.maximum.accumulate(pos_passed_fuel_mask_full, axis=0)

            if not is_last_iteration:
                furthest_out = np.maximum.accumulate(x[:, None])
                neg_passed_fuel_mask11 = (x[:, None] <= fuel_x[None]) & is_stopped & (furthest_out > fuel_x[None])
                neg_passed_fuel_mask = np.maximum.accumulate(neg_passed_fuel_mask11, axis=0)
            else:
                neg_passed_fuel_mask = np.zeros_like(pos_passed_fuel_mask)

            withdrawn_fuels_multiplier = np.array(pos_passed_fuel_mask, dtype=np.int32) + np.array(neg_passed_fuel_mask, dtype=np.int32)
            withdrawn_fuels_volume = withdrawn_fuels_multiplier * withdraw_volumes
            total_withdrawn_fuels_volume = np.sum(withdrawn_fuels_volume, axis=-1)

            net_fuel = total_withdrawn_fuels_volume - np.array(deposited_fuel)

            fuel = 1 - x_covered + net_fuel
            fuels.extend(fuel)

            net_fuel_changes = np.diff(net_fuel, axis=0)
            net_fuel_change_indices = np.nonzero(net_fuel_changes)

            fuel_changes.extend(zip(start_point + net_fuel_change_indices[0] + 1, net_fuel_changes[net_fuel_change_indices]))

        self.plot_visulization(np.array(xs), np.array(fuels), np.array(trips), fuel_changes, fps, paus_time, do_animation)

    def plot_visulization(self, xs, fuels, trips, fuel_changes, fps, paus_time, do_animation):
        use_car_add = 1 if do_animation else 0

        fig, axs = plt.subplots(len(trips) + use_car_add, figsize=(7, 0.5 + 1.5 * (len(trips) + use_car_add)), sharex=True)
        if len(trips) + use_car_add == 1:
            axs = [axs]  # Why does matplotlib do this?

        if use_car_add:
            axs[0].set_xlim(0, np.max(xs))
            height = 1.5
            axs[0].set_ylim(0, height)
            axs[0].set_yticks([])

            point, = axs[0].plot([], [], "bo", markersize=8, label="Jeep")
            text = axs[0].text(0, 0, "", ha="center", va="bottom", fontsize=10, color="orange")
            text2 = axs[0].text(0, 0, "", ha="center", va="bottom", fontsize=10)

        lines = []
        for i, trip in enumerate(trips):
            line = axs[i + use_car_add].plot([], [], color="orange")
            lines.append(line[0])

            axs[i + use_car_add].set_xlim(0, np.max(xs))
            axs[i + use_car_add].set_ylim(0, 1)
            axs[i + use_car_add].set_title(f"Trip {i + 1}")
            axs[i + use_car_add].set_ylabel("Fuel Level")

        plt.xlabel("Displacement")
        plt.tight_layout()

        def animate(i):
            index = index_script[i]

            all_trips = np.array(trips)
            trip_index = int(np.nonzero((index >= all_trips[:, 0]) & (index < all_trips[:, 2]))[0][0])
            for trip_i, trip in enumerate(all_trips[:trip_index]):
                lines[trip_i].set_data(xs[trip[0]:trip[2] + 1], fuels[trip[0]:trip[2] + 1])

            start_index = int(all_trips[trip_index][0])
            lines[trip_index].set_data(xs[start_index:index + 1], fuels[start_index:index + 1])

            # print("Animating frame", i, "index:", index)
            if not use_car_add:
                return

            x = xs[index]
            point.set_data([x], [height / 2])
            text.set_position((x, height / 2 + 0.1))

            np_fuel_change = np.array(fuel_changes).reshape(-1, 2)
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

        index_script += list(range(fuel_changes[-1][0] if len(fuel_changes) else 0, len(xs))) + [len(xs) - 1] * paus_frames
        # """

        # Unecessary to create entire script if not do_animation but whatever

        if not do_animation:
            animate(len(index_script) - 1)

            image_path = os.path.join(self.images_path, f"image_{len(trips)}trips.png")
            plt.savefig(image_path)

            return

        ani = animation.FuncAnimation(fig, animate, frames=len(index_script))
        animation_path = os.path.join(self.animations_path, f"animation_{len(trips)}trips.gif")
        ani.save(animation_path, fps=fps)


if __name__ == "__main__":
    solver = JeepSolver()
    solver.simulate_trip()
    solver.visualize_trip()
