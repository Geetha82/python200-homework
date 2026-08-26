# Week 8 Project: Cloud Infrastructure Setup & Cost Analysis

* **Course:** Python 200
* **Submission Date:** August 24, 2026
* **Project Video Walkthrough:** [Watch Video Walkthrough](https://youtu.be/WmwmFeixCTE)
* **Live AWS Estimate Link:** [View My AWS Pricing Calculator Estimate]
(https://calculator.aws/#/estimate?id=243d23ff685ffa4c35fbb9a6a552f4aff0ac3245)

-# Project 08: Supabase Setup and Cloud Cost Analysis

## Part A: Supabase Project Setup

### Confirmation Statement
The Supabase project **python200** has been successfully provisioned and configured. The local environment files (`.env` and `.gitignore`) have been structured to securely store the project credentials without exposing them to public version control. 

Through the Table Editor, I have verified that both required tables are properly created and structured with the correct schemas:
* **`weather_raw`**: Confirmed fields include `date` (Primary Key), `temperature_2m_max`, `temperature_2m_min`, `precipitation_sum`, `wind_speed_10m_max`, and `loaded_at`.
* **`weather_enriched`**: Confirmed fields include `date` (Foreign Key referencing `weather_raw`), `good_for_running`, `confidence`, `llm_summary`, and `enriched_at`.

Row Level Security (RLS) has been successfully disabled on both tables to allow seamless data pipelines for the upcoming work in weeks 9-11. No issues or blocks were encountered during this setup phase.

---

## Part B: Cloud Cost Analysis

### Infrastructure Cost Breakdown
Using the AWS Pricing Calculator with the **US East (N. Virginia)** region, the infrastructure configurations yield the following monthly costs:

* **Scenario A (Lightweight Compute):** 
  * **Service:** Amazon EC2 (`t3.micro`, 1 vCPU, 1 GB RAM)
  * **Workload:** Part-time scheduling (160 hours/month)
  * **Monthly Cost:** **$7.23 USD**
* **Scenario B (Heavy Analytics Workload):**
  * **Compute:** Amazon EC2 (`p3.2xlarge`, 8 vCPU, 1 V100 GPU running 24/7 for 730 hours) $\rightarrow$ **$2,233.80 USD**
  * **Database:** Amazon RDS for PostgreSQL (`db.m5.large` instance with 100 GB gp2 storage) $\rightarrow$ **$159.69 USD**
  * **Storage:** Amazon S3 Standard (1 TB storage allocation per month) $\rightarrow$ **$23.55 USD**
  * **Scenario B Subtotal:** **$2,417.04 USD**

### Final Estimate Summary
* **Total Monthly Budget:** **$2,424.27 USD**
* **Total Annual Cost (12 Months):** **$29,091.24 USD**

### Reflection & Surprises
The extreme cost difference between these two scenarios highlights how quickly cloud expenditures scale depending on compute choices. What stands out most is how heavily specialized GPU compute dominates the monthly budget. In Scenario B, the single `p3.2xlarge` instance accounts for over **92%** of the entire bill ($2,233.80 out of $2,424.27). By comparison, provisioning a robust 100 GB production-ready PostgreSQL database and a massive 1 TB file storage bucket looks financially trivial. 

### Additional Calculator Exploration
While exploring the calculator beyond the baseline requirements, I noticed that changing storage variants and availability tiers dramatically alters operational costs. For example, moving from General Purpose SSDs (gp2) to Provisioned IOPS (io2) or scaling the RDS database to a Multi-AZ (Availability Zone) deployment for high availability quickly doubles database costs. Furthermore, because these estimates assume an "On-Demand" pricing strategy, implementing AWS Compute Savings Plans or utilizing Spot Instances for the analytics workloads could drastically lower the baseline costs by 30% to 70%.

### GPU Trade-offs & Economic Feasibility
Comparing these two scenarios demonstrates that a high-tier GPU instance is strictly an acceleration tool meant for demanding parallel processing tasks—such as training deep learning models or executing heavy computer vision pipelines. Running a GPU instance continuously (730 hours/month) introduces a punishing financial penalty. If an application's requirements are sparse or its analytical algorithms can execute effectively on standard CPUs, sticking with instances similar to the lightweight `t3` family is exponentially more economical. GPU instances are only worth their steep costs when the reduction in processing time directly creates clear business or operational value.
