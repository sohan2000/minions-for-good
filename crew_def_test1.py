# crew_definition.py
from crewai import Agent, Crew, Task

class AllAgentsCrew:
    """Multi‑agent crew that gathers multiple quotes (2‑sentence limit per agent)."""
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.crew = self._create_crew()

    # ---------- helpers ----------
    def _mk_company(self, role, backstory):
        return Agent(
            role=role,
            goal="Provide a concise, two‑sentence quote for its trade.",
            backstory=backstory,
            verbose=self.verbose,
        )

    # ---------- crew ----------
    def _create_crew(self) -> Crew:
        builder = Agent(
            role="Builder",
            goal="Coordinate suppliers and deliver the best overall deal",
            backstory="Veteran general contractor; summarises quotes & picks winners.",
            verbose=self.verbose,
        )

        client = Agent(
            role="Client",
            goal="Approve or reject the builder’s bundled bid, then wait for milestone reports.",
            backstory="Acts as the project owner and funding source.",
            verbose=self.verbose,
        )


        # suppliers
        construction_A = self._mk_company("ConstructionCo‑A", "Large firm with bulk discounts")
        construction_B = self._mk_company("ConstructionCo‑B", "Fast‑schedule medium builder")
        construction_C = self._mk_company("ConstructionCo‑C", "Boutique cost‑focused builder")

        wood_A = self._mk_company("WoodworkCo‑A", "Premium hardwood specialist")
        wood_B = self._mk_company("WoodworkCo‑B", "Budget‑friendly engineered‑wood shop")

        paint_A = self._mk_company("PaintCo‑A", "Eco‑friendly painter with 10‑year warranty")
        paint_B = self._mk_company("PaintCo‑B", "High‑volume, low‑cost painter")

        # tasks
        tasks = [
            Task(
                description=(
                    "Parse the client brief: {input_data}. Extract budget (+10 % stretch), "
                    "deadline and key preferences. Reply in **≤2 sentences**."
                ),
                expected_output="Req brief in 2 sentences",
                agent=builder,
            ),
            # construction quotes
            *[
                Task(
                    description="Give your best construction quote in **≤2 sentences**.",
                    expected_output=f"Construction quote {suffix} (2 sentences)",
                    agent=agent,
                )
                for suffix, agent in zip(
                    ["A", "B", "C"],
                    [construction_A, construction_B, construction_C],
                )
            ],
            # woodwork quotes
            *[
                Task(
                    description="Give your best woodwork quote in **≤2 sentences**.",
                    expected_output=f"Woodwork quote {suffix} (2 sentences)",
                    agent=agent,
                )
                for suffix, agent in zip(["A", "B"], [wood_A, wood_B])
            ],
            # paint quotes
            *[
                Task(
                    description="Give your best paint quote in **≤2 sentences**.",
                    expected_output=f"Paint quote {suffix} (2 sentences)",
                    agent=agent,
                )
                for suffix, agent in zip(["A", "B"], [paint_A, paint_B])
            ],
            # builder chooses
            Task(
                description=(
                    "Compare all quotes against budget & preferences. Pick one construction, "
                    "one woodwork and one paint company. Explain each quote **and** why the "
                    "combination is chosen. No strict length limit here."
                ),
                expected_output=(
                    "Final deal: chosen suppliers, combined cost/duration, justification."
                ),
                agent=builder,
            ),
            Task(  # 1‑line acceptance / rejection
                description="Reply **ACCEPT** or **REJECT** in one sentence.",
                expected_output="ACCEPT | REJECT + 1‑sentence comment",
                agent=client,
            ),
            Task(  # kick‑off
                description="If the client accepted, notify all chosen suppliers to start work.",
                expected_output="Work‑order sent to each winning supplier.",
                agent=builder,
            ),
            Task(  # progress aggregation (repeat this block per milestone if you like)
                description="Collect progress % and money spent from every active supplier, "
                            "summarise in JSON {{supplier, pct_complete, dollars_spent}}.",
                expected_output="ProgressSummary‑M1‑JSON",
                agent=builder,
            ),
            Task(  # payment release & log
                description="When a supplier hits its milestone (pct_complete ≥ 100 for M1), "
                            "log a **PaymentLog‑M1** entry and mark that milestone closed.",
                expected_output="PaymentLog‑M1 entry",
                agent=builder,
            ),
            Task(
                description="Verify every supplier shows pct_complete == 100. "
                            "If true, declare the house finished and output a final ledger "
                            "of all PaymentLog entries.",
                expected_output="ProjectComplete + FinalLedger JSON",
                agent=builder,
            )
        ]

        agents = [
            client,
            builder,
            construction_A, construction_B, construction_C,
            wood_A, wood_B,
            paint_A, paint_B,
        ]
        return Crew(agents=agents, tasks=tasks)
