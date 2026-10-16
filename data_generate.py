from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("financial_dataset")

# Number of rows in each fact table
FACT_ROWS = 5_100_000

# Number of rows generated per batch
CHUNK_SIZE = 250_000

# Reproducible random seed
SEED = 42

# Date range
START_DATE = "2020-01-01"
END_DATE = "2026-12-31"

# Number of dimension members
N_CATEGORIES = 30
N_DEPARTMENTS = 20
N_PRODUCTS = 500
N_REGIONS = 15
N_COST_CENTERS = 100
N_CUSTOMERS = 100_000
N_SUPPLIERS = 30_000


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(SEED)


# ============================================================
# MASTER DATA
# ============================================================

CATEGORIES = [
    "Sales",
    "Marketing",
    "Operations",
    "Technology",
    "Human Resources",
    "Finance",
    "Logistics",
    "Procurement",
    "Manufacturing",
    "Research and Development",
    "Customer Service",
    "Legal",
    "Administration",
    "Facilities",
    "Travel",
    "Training",
    "Insurance",
    "Utilities",
    "Taxes",
    "Consulting",
    "Maintenance",
    "Security",
    "Software",
    "Hardware",
    "Professional Services",
    "Transportation",
    "Communication",
    "Rent",
    "Banking",
    "Other",
]

SUBCATEGORIES = {
    "Sales": [
        "Product Sales",
        "Service Sales",
        "Subscriptions",
        "Licensing",
    ],
    "Marketing": [
        "Advertising",
        "Events",
        "Digital Marketing",
        "Content",
    ],
    "Operations": [
        "Operational Expenses",
        "Supplies",
        "Equipment",
        "Services",
    ],
    "Technology": [
        "Cloud",
        "Software",
        "Hardware",
        "Infrastructure",
    ],
    "Human Resources": [
        "Payroll",
        "Benefits",
        "Recruitment",
        "Training",
    ],
    "Finance": [
        "Bank Fees",
        "Accounting",
        "Audit",
        "Financial Services",
    ],
    "Logistics": [
        "Freight",
        "Shipping",
        "Warehousing",
        "Transportation",
    ],
    "Procurement": [
        "Raw Materials",
        "Office Supplies",
        "Equipment",
        "Services",
    ],
    "Manufacturing": [
        "Raw Materials",
        "Production",
        "Maintenance",
        "Quality Control",
    ],
    "Research and Development": [
        "Research",
        "Development",
        "Testing",
        "Innovation",
    ],
    "Customer Service": [
        "Support",
        "Call Center",
        "Customer Success",
        "Returns",
    ],
    "Legal": [
        "Legal Services",
        "Contracts",
        "Compliance",
        "Litigation",
    ],
    "Administration": [
        "Office",
        "Supplies",
        "Administration Services",
        "Management",
    ],
    "Facilities": [
        "Rent",
        "Utilities",
        "Cleaning",
        "Repairs",
    ],
    "Travel": [
        "Flights",
        "Hotels",
        "Meals",
        "Transportation",
    ],
    "Training": [
        "Courses",
        "Certifications",
        "Workshops",
        "Conferences",
    ],
    "Insurance": [
        "Property",
        "Health",
        "Liability",
        "Vehicle",
    ],
    "Utilities": [
        "Electricity",
        "Water",
        "Gas",
        "Internet",
    ],
    "Taxes": [
        "Income Tax",
        "Sales Tax",
        "Property Tax",
        "Other Taxes",
    ],
    "Consulting": [
        "Management",
        "Technology",
        "Finance",
        "Strategy",
    ],
    "Maintenance": [
        "Equipment",
        "Buildings",
        "Vehicles",
        "Systems",
    ],
    "Security": [
        "Physical Security",
        "Cybersecurity",
        "Monitoring",
        "Access Control",
    ],
    "Software": [
        "Licenses",
        "Subscriptions",
        "Development",
        "Support",
    ],
    "Hardware": [
        "Computers",
        "Servers",
        "Networking",
        "Peripherals",
    ],
    "Professional Services": [
        "Legal",
        "Accounting",
        "Consulting",
        "Advisory",
    ],
    "Transportation": [
        "Fleet",
        "Fuel",
        "Public Transportation",
        "Car Rental",
    ],
    "Communication": [
        "Telephone",
        "Internet",
        "Postal",
        "Media",
    ],
    "Rent": [
        "Office",
        "Warehouse",
        "Equipment",
        "Vehicles",
    ],
    "Banking": [
        "Bank Fees",
        "Interest",
        "Loans",
        "Transfers",
    ],
    "Other": [
        "Other Expenses",
        "Other Revenue",
        "Miscellaneous",
        "Adjustments",
    ],
}

DEPARTMENTS = [
    "Finance",
    "Sales",
    "Marketing",
    "Operations",
    "Human Resources",
    "IT",
    "Procurement",
    "Logistics",
    "Manufacturing",
    "Research and Development",
    "Customer Service",
    "Legal",
    "Administration",
    "Facilities",
    "Executive",
    "Quality",
    "Security",
    "Compliance",
    "Strategy",
    "Business Development",
]

REGIONS = [
    "North",
    "South",
    "Southeast",
    "Northeast",
    "Central-West",
    "International",
    "North America",
    "South America",
    "Europe",
    "Asia",
    "Africa",
    "Oceania",
    "Middle East",
    "Central America",
    "Caribbean",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def random_dates(size, start=START_DATE, end=END_DATE):
    """Generate random dates between two dates."""

    start_ts = pd.Timestamp(start).value // 10**9
    end_ts = pd.Timestamp(end).value // 10**9

    timestamps = rng.integers(
        start_ts,
        end_ts,
        size=size,
    )

    return pd.to_datetime(timestamps, unit="s")


def random_amounts(size, minimum=10, maximum=100000):
    """Generate realistic financial amounts."""

    values = rng.lognormal(
        mean=np.log(3000),
        sigma=1.1,
        size=size,
    )

    values = np.clip(
        values,
        minimum,
        maximum,
    )

    return np.round(values, 2)


def random_choice(values, size):
    """Randomly select values from an array."""

    return rng.choice(
        values,
        size=size,
    )


# ============================================================
# DIMENSION: CATEGORY
# ============================================================

def create_dim_categoria():
    data = []

    for i, category in enumerate(CATEGORIES, start=1):
        subcategories = SUBCATEGORIES.get(
            category,
            ["General"],
        )

        for subcategory in subcategories:
            data.append(
                {
                    "Category_ID": len(data) + 1,
                    "Category": category,
                    "Subcategory": subcategory,
                }
            )

    return pd.DataFrame(data)


# ============================================================
# DIMENSION: DEPARTMENT
# ============================================================

def create_dim_departamento():
    return pd.DataFrame(
        {
            "Department_ID": np.arange(
                1,
                len(DEPARTMENTS) + 1,
            ),
            "Department": DEPARTMENTS,
        }
    )


# ============================================================
# DIMENSION: PRODUCT
# ============================================================

def create_dim_produto():
    product_ids = np.arange(
        1,
        N_PRODUCTS + 1,
    )

    product_names = [
        f"Product_{i:05d}"
        for i in product_ids
    ]

    return pd.DataFrame(
        {
            "Product_ID": product_ids,
            "Product": product_names,
            "Product_Type": rng.choice(
                [
                    "Physical",
                    "Digital",
                    "Service",
                    "Subscription",
                ],
                size=N_PRODUCTS,
            ),
            "Product_Group": rng.choice(
                [
                    "Group A",
                    "Group B",
                    "Group C",
                    "Group D",
                    "Group E",
                ],
                size=N_PRODUCTS,
            ),
        }
    )


# ============================================================
# DIMENSION: REGION
# ============================================================

def create_dim_regiao():
    return pd.DataFrame(
        {
            "Region_ID": np.arange(
                1,
                len(REGIONS) + 1,
            ),
            "Region": REGIONS,
        }
    )


# ============================================================
# DIMENSION: COST CENTER
# ============================================================

def create_dim_centro_custo():
    ids = np.arange(
        1,
        N_COST_CENTERS + 1,
    )

    return pd.DataFrame(
        {
            "Cost_Center_ID": ids,
            "Cost_Center": [
                f"CC-{i:04d}"
                for i in ids
            ],
            "Cost_Center_Name": [
                f"Cost Center {i:04d}"
                for i in ids
            ],
        }
    )


# ============================================================
# DIMENSION: CUSTOMER
# ============================================================

def create_dim_cliente():
    ids = np.arange(
        1,
        N_CUSTOMERS + 1,
    )

    return pd.DataFrame(
        {
            "Customer_ID": ids,
            "Customer": [
                f"Customer_{i:06d}"
                for i in ids
            ],
            "Customer_Type": rng.choice(
                [
                    "Individual",
                    "Small Business",
                    "Medium Business",
                    "Enterprise",
                ],
                size=N_CUSTOMERS,
            ),
            "Region_ID": rng.integers(
                1,
                len(REGIONS) + 1,
                size=N_CUSTOMERS,
            ),
        }
    )


# ============================================================
# DIMENSION: SUPPLIER
# ============================================================

def create_dim_fornecedor():
    ids = np.arange(
        1,
        N_SUPPLIERS + 1,
    )

    return pd.DataFrame(
        {
            "Supplier_ID": ids,
            "Supplier": [
                f"Supplier_{i:05d}"
                for i in ids
            ],
            "Supplier_Type": rng.choice(
                [
                    "Local",
                    "National",
                    "International",
                    "Strategic",
                ],
                size=N_SUPPLIERS,
            ),
            "Region_ID": rng.integers(
                1,
                len(REGIONS) + 1,
                size=N_SUPPLIERS,
            ),
        }
    )


# ============================================================
# DIMENSION: CALENDAR
# ============================================================

def create_dim_calendario():
    dates = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="D",
    )

    df = pd.DataFrame(
        {
            "Date": dates,
        }
    )

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.month_name()
    df["Quarter"] = df["Date"].dt.quarter
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["Day_of_Week"] = df["Date"].dt.dayofweek + 1
    df["Day_Name"] = df["Date"].dt.day_name()
    df["Is_Weekend"] = df["Day_of_Week"] >= 6

    return df


# ============================================================
# FACT TABLE: TRANSACTIONS
# ============================================================

def generate_transactions_chunk(
    start_id,
    size,
):
    transaction_ids = np.arange(
        start_id,
        start_id + size,
    )

    transaction_dates = random_dates(size)

    transaction_type = rng.choice(
        [
            "Revenue",
            "Expense",
        ],
        size=size,
        p=[
            0.45,
            0.55,
        ],
    )

    categories = random_choice(
        CATEGORIES,
        size,
    )

    subcategories = [
        rng.choice(
            SUBCATEGORIES.get(
                category,
                ["General"],
            )
        )
        for category in categories
    ]

    departments = random_choice(
        DEPARTMENTS,
        size,
    )

    products = rng.integers(
        1,
        N_PRODUCTS + 1,
        size=size,
    )

    regions = rng.integers(
        1,
        len(REGIONS) + 1,
        size=size,
    )

    cost_centers = rng.integers(
        1,
        N_COST_CENTERS + 1,
        size=size,
    )

    customers = rng.integers(
        1,
        N_CUSTOMERS + 1,
        size=size,
    )

    suppliers = rng.integers(
        1,
        N_SUPPLIERS + 1,
        size=size,
    )

    status = rng.choice(
        [
            "Paid",
            "Pending",
            "Received",
        ],
        size=size,
        p=[
            0.65,
            0.20,
            0.15,
        ],
    )

    return pd.DataFrame(
        {
            "Transaction_ID": transaction_ids,
            "Date": transaction_dates,
            "Type": transaction_type,
            "Category": categories,
            "Subcategory": subcategories,
            "Amount": random_amounts(size),
            "Department": departments,
            "Product_ID": products,
            "Region_ID": regions,
            "Cost_Center_ID": cost_centers,
            "Status": status,
            "Customer_ID": customers,
            "Supplier_ID": suppliers,
        }
    )


# ============================================================
# FACT TABLE: ACCOUNTS PAYABLE
# ============================================================

def generate_accounts_payable_chunk(
    start_id,
    size,
):
    due_dates = random_dates(size)

    status = rng.choice(
        [
            "Paid",
            "Pending",
            "Overdue",
        ],
        size=size,
        p=[
            0.60,
            0.25,
            0.15,
        ],
    )

    return pd.DataFrame(
        {
            "Account_ID": np.arange(
                start_id,
                start_id + size,
            ),
            "Supplier_ID": rng.integers(
                1,
                N_SUPPLIERS + 1,
                size=size,
            ),
            "Category": random_choice(
                CATEGORIES,
                size,
            ),
            "Due_Date": due_dates,
            "Amount": random_amounts(size),
            "Status": status,
            "Department": random_choice(
                DEPARTMENTS,
                size,
            ),
        }
    )


# ============================================================
# FACT TABLE: ACCOUNTS RECEIVABLE
# ============================================================

def generate_accounts_receivable_chunk(
    start_id,
    size,
):
    due_dates = random_dates(size)

    status = rng.choice(
        [
            "Received",
            "Pending",
            "Overdue",
        ],
        size=size,
        p=[
            0.65,
            0.20,
            0.15,
        ],
    )

    return pd.DataFrame(
        {
            "Account_ID": np.arange(
                start_id,
                start_id + size,
            ),
            "Customer_ID": rng.integers(
                1,
                N_CUSTOMERS + 1,
                size=size,
            ),
            "Due_Date": due_dates,
            "Amount": random_amounts(size),
            "Status": status,
            "Region_ID": rng.integers(
                1,
                len(REGIONS) + 1,
                size=size,
            ),
        }
    )


# ============================================================
# FACT TABLE: BUDGET
# ============================================================

def generate_budget():
    years = np.arange(
        2020,
        2027,
    )

    months = np.arange(
        1,
        13,
    )

    rows = []

    budget_id = 1

    for year in years:
        for month in months:

            for category_id in range(
                1,
                len(CATEGORIES) + 1,
            ):

                department_id = rng.integers(
                    1,
                    len(DEPARTMENTS) + 1,
                )

                product_id = rng.integers(
                    1,
                    N_PRODUCTS + 1,
                )

                region_id = rng.integers(
                    1,
                    len(REGIONS) + 1,
                )

                cost_center_id = rng.integers(
                    1,
                    N_COST_CENTERS + 1,
                )

                rows.append(
                    {
                        "Budget_ID": budget_id,
                        "Year": year,
                        "Month": month,
                        "Category_ID": category_id,
                        "Department_ID": department_id,
                        "Product_ID": product_id,
                        "Region_ID": region_id,
                        "Cost_Center_ID": cost_center_id,
                        "Budget": round(
                            float(
                                rng.uniform(
                                    10000,
                                    500000,
                                )
                            ),
                            2,
                        ),
                    }
                )

                budget_id += 1

    return pd.DataFrame(rows)


# ============================================================
# WRITE DIMENSIONS
# ============================================================

def save_dimensions():
    print("\nCreating dimension tables...")

    dimensions = {
        "Dim_Calendario": create_dim_calendario(),
        "Dim_Categoria": create_dim_categoria(),
        "Dim_Departamento": create_dim_departamento(),
        "Dim_Produto": create_dim_produto(),
        "Dim_Regiao": create_dim_regiao(),
        "Dim_Centro_Custo": create_dim_centro_custo(),
        "Dim_Cliente": create_dim_cliente(),
        "Dim_Fornecedor": create_dim_fornecedor(),
    }

    for name, dataframe in dimensions.items():

        output_file = OUTPUT_DIR / f"{name}.parquet"

        dataframe.to_parquet(
            output_file,
            index=False,
            compression="snappy",
        )

        print(
            f"{name}: "
            f"{len(dataframe):,} rows"
        )


# ============================================================
# WRITE BUDGET
# ============================================================

def save_budget():
    print("\nCreating Budget table...")

    budget = generate_budget()

    output_file = OUTPUT_DIR / "Budget.parquet"

    budget.to_parquet(
        output_file,
        index=False,
        compression="snappy",
    )

    print(
        f"Budget: "
        f"{len(budget):,} rows"
    )


# ============================================================
# GENERIC CHUNK WRITER
# ============================================================

def generate_fact_table(
    table_name,
    generator_function,
):
    print(
        f"\nGenerating {table_name}..."
    )

    output_file = (
        OUTPUT_DIR /
        f"{table_name}.parquet"
    )

    first_chunk = True

    total_generated = 0

    while total_generated < FACT_ROWS:

        remaining = FACT_ROWS - total_generated

        current_chunk_size = min(
            CHUNK_SIZE,
            remaining,
        )

        start_id = total_generated + 1

        dataframe = generator_function(
            start_id,
            current_chunk_size,
        )

        if first_chunk:

            dataframe.to_parquet(
                output_file,
                index=False,
                compression="snappy",
            )

            first_chunk = False

        else:

            # PyArrow is used to append subsequent chunks.
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(
                dataframe,
                preserve_index=False,
            )

            parquet_writer.write_table(table)

        total_generated += current_chunk_size

        print(
            f"{table_name}: "
            f"{total_generated:,} / "
            f"{FACT_ROWS:,}"
        )

    print(
        f"Finished {table_name}: "
        f"{total_generated:,} rows"
    )


# ============================================================
# BETTER PARQUET CHUNK WRITER
# ============================================================

def generate_fact_table_parquet(
    table_name,
    generator_function,
):
    print(
        f"\nGenerating {table_name}..."
    )

    output_file = (
        OUTPUT_DIR /
        f"{table_name}.parquet"
    )

    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None

    total_generated = 0

    try:

        while total_generated < FACT_ROWS:

            remaining = (
                FACT_ROWS -
                total_generated
            )

            current_chunk_size = min(
                CHUNK_SIZE,
                remaining,
            )

            start_id = (
                total_generated + 1
            )

            dataframe = generator_function(
                start_id,
                current_chunk_size,
            )

            table = pa.Table.from_pandas(
                dataframe,
                preserve_index=False,
            )

            if writer is None:

                writer = pq.ParquetWriter(
                    output_file,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)

            total_generated += (
                current_chunk_size
            )

            print(
                f"  {total_generated:,} / "
                f"{FACT_ROWS:,}"
            )

    finally:

        if writer is not None:
            writer.close()

    print(
        f"Finished {table_name}: "
        f"{total_generated:,} rows"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = datetime.now()

    print("=" * 70)
    print("LARGE-SCALE FINANCIAL DATASET GENERATOR")
    print("=" * 70)

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print(
        f"Rows per fact table: "
        f"{FACT_ROWS:,}"
    )

    print(
        f"Chunk size: "
        f"{CHUNK_SIZE:,}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    save_dimensions()

    # --------------------------------------------------------
    # Budget
    # --------------------------------------------------------

    save_budget()

    # --------------------------------------------------------
    # Fact tables
    # --------------------------------------------------------

    generate_fact_table_parquet(
        "Fato_Transacoes",
        generate_transactions_chunk,
    )

    generate_fact_table_parquet(
        "Contas_Pagar",
        generate_accounts_payable_chunk,
    )

    generate_fact_table_parquet(
        "Contas_Receber",
        generate_accounts_receivable_chunk,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    elapsed = (
        datetime.now() -
        start_time
    )

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETED")
    print("=" * 70)

    print(
        f"Execution time: {elapsed}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print("\nGenerated files:")

    for file in sorted(
        OUTPUT_DIR.glob("*.parquet")
    ):
        size_mb = (
            file.stat().st_size /
            1024 /
            1024
        )

        print(
            f"  {file.name:<30} "
            f"{size_mb:,.2f} MB"
        )


if __name__ == "__main__":
    main()