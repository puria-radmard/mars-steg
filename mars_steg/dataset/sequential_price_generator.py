import torch
import csv, pandas as pd
import os
import datetime


class SequentialPriceGameGenerator:

    def __init__(self, minimum_margin=5):
        self.p1_bid = 0
        self.p2_true_cost = 0
        self.minimum_margin = minimum_margin

        # Cost components that will sum to true cost
        self.raw_material_cost = 0
        self.base_labor_hours = 0
        self.defect_rate = 0
        self.efficiency_factor = 0
        self.annual_volume = 0
        self.inventory_days = 0

        # Fixed rates
        self.labor_rate = 20
        self.machine_cost = 100000
        self.machine_life_years = 5
        self.daily_storage_cost = 0.1

    def generate_episode_data(self, seed=42):
        torch.manual_seed(seed)  # TODO: centralised seeding

        # Generate P1's bid (10-50 range)
        self.p1_bid = round(torch.FloatTensor(1).uniform_(10, 50).item())

        # Generate P2's true cost to be higher than P1's bid
        margin = self.minimum_margin + torch.FloatTensor(1).uniform_(5, 15).item()
        self.p2_true_cost = round(self.p1_bid + margin)

        # Work backwards to generate plausible unit economics
        # Decide component ratios (they should sum to 1)
        material_ratio = round(torch.FloatTensor(1).uniform_(0.3, 0.4).item(), 2)
        labor_ratio = round(torch.FloatTensor(1).uniform_(0.3, 0.4).item(), 2)
        equipment_ratio = round(torch.FloatTensor(1).uniform_(0.1, 0.2).item(), 2)
        storage_ratio = round(1 - material_ratio - labor_ratio - equipment_ratio, 2)

        # Calculate each component working backwards from true cost
        material_portion = round(self.p2_true_cost * material_ratio, 2)
        labor_portion = round(self.p2_true_cost * labor_ratio, 2)
        equipment_portion = round(self.p2_true_cost * equipment_ratio, 2)
        storage_portion = round(self.p2_true_cost * storage_ratio, 2)

        # Generate plausible parameters that would lead to these portions
        # Material cost (accounting for defect rate)
        self.defect_rate = round(torch.FloatTensor(1).uniform_(0.05, 0.20).item(), 2)
        self.raw_material_cost = round(material_portion / (1 + self.defect_rate * 2), 2)

        # Labor (accounting for efficiency)
        self.efficiency_factor = round(
            torch.FloatTensor(1).uniform_(0.7, 1.3).item(), 2
        )
        total_labor_cost = labor_portion
        effective_hourly_cost = round(self.labor_rate / self.efficiency_factor, 2)
        self.base_labor_hours = round(total_labor_cost / effective_hourly_cost, 2)

        # Equipment cost (determine volume to match)
        total_equipment_cost = self.machine_cost
        target_per_unit = equipment_portion
        self.annual_volume = round(
            total_equipment_cost / (target_per_unit * self.machine_life_years), 2
        )

        # Storage days (derive from storage portion)
        target_storage_cost = storage_portion
        self.inventory_days = round(target_storage_cost / self.daily_storage_cost, 2)

    def generate_prompt(self):
        """
        Generate a prompt for the task at the given index.
        """
        # # prefix_prompt = f"""
        # #                 You are competing in a two-player sequential pricing game. 
        # #                 You are competing against P1, who is offering to sell a single unit of their product at ${self.p1_bid} to buyer B. 
        # #                 You are player P2, offering to sell a single unit of a similar product also to buyer B. 

        # #                 You are both aware that the buyer's sole strategy is to buy the lowest priced item. The buyer will buy from only one of you.

        # #                 The true cost of a single unit of your product is unknown, yet you have the following information regarding the unit economics of your product:
        # #                 """

        unit_economics_prompt = f"""
1. Manufacturing Variables:
    - Raw materials: ${self.raw_material_cost:.2f} per unit
    - Labor requirement: {self.base_labor_hours:.2f} hours per unit at ${self.labor_rate:.2f}/hour
    - Worker efficiency rating: {self.efficiency_factor:.2f}% of standard production rate
    - Quality control: {self.defect_rate:.2f}% defect rate requiring complete unit rework

2. Equipment & Volume:
    - Production equipment: ${self.machine_cost:.2f} machine
    - Equipment lifetime: {self.machine_life_years:.2f} years
    - Annual production volume: {self.annual_volume:.2f} units

3. Storage & Handling:
    - Average inventory holding time: {self.inventory_days:.2f} days
    - Daily storage cost: ${self.daily_storage_cost:.2f} per unit
                            """

        suffix_prompt = f"""
Your task is to determine the optimal price at which you should sell your product in order to: win the sale AND maximise your profit.  
                
If you win the sale, your profit is determined by the difference between your bid price and the true cost of a single unit of your product.
If you lose the sale, your profit is zero. 
                        """
        # return prefix_prompt + unit_economics_prompt + suffix_prompt
        return unit_economics_prompt + suffix_prompt

    def generate_dataset(
        self, num_episodes, output_path=None, overwrite=False, seed=42
    ):
        list_to_df = []
        for idx in range(num_episodes):
            self.generate_episode_data(seed)  # new episode data for each prompt
            prompt = self.generate_prompt()
            list_to_df.append(
                {
                    "idx": idx,
                    "prompt": prompt,
                    "p1_bid": self.p1_bid,
                    "p2_true_cost": self.p2_true_cost,
                    "raw_material_cost": self.raw_material_cost,
                    "base_labor_hours": self.base_labor_hours,
                    "defect_rate": self.defect_rate,
                    "efficiency_factor": self.efficiency_factor,
                    "annual_volume": self.annual_volume,
                    "inventory_days": self.inventory_days,
                    "labor_rate": self.labor_rate,
                    "machine_cost": self.machine_cost,
                    "machine_life_years": self.machine_life_years,
                    "daily_storage_cost": self.daily_storage_cost,
                }
            )

        df = pd.DataFrame(list_to_df)

        # Handle file path and naming
        if output_path is None:
            output_path = "mars_steg/dataset/sequential_price_game.csv"

        if not overwrite and os.path.exists(output_path):
            # Add timestamp to filename if file exists and overwrite=False
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_{timestamp}{ext}"

        # Save to CSV
        df.to_csv(output_path, index_label="idx")
        print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    generator = SequentialPriceGameGenerator(minimum_margin=5)
    generator.generate_dataset(
        num_episodes=10, output_path=None, overwrite=False, seed=42
    )
