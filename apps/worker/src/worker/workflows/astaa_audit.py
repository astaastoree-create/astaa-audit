import pydantic
import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai import Agent, Runner
from mistralai.workflows.plugins.mistralai.connectors import connector, uses_connectors

ASTAA_CONNECTOR_NAME = "astaa_7e0d"
astaa_connector = connector(ASTAA_CONNECTOR_NAME)

AUDIT_INSTRUCTIONS = """You are auditing the checkout funnel of the ASTAA Shopify store (astaa.store)
through the UCP/MCP connector tools available to you.

Hard rules:
- Never call complete_checkout or any tool that submits payment or places an order.
- Only use catalog/search/product/checkout-creation read-and-prepare tools.
- If a tool fails or returns messages, record the exact message text and continue.

Steps:
1. Use search_catalog to find a low-priced, available item. Record the product id, variant id,
   title, price and availability. If search_catalog fails, try get_product with a product URL
   or handle from https://astaa.store.
2. Create a checkout with that single variant (quantity 1) using a United States test address
   context (any realistic city/state/zip, currency USD). Then fetch the checkout with
   get_checkout (or the create response).
3. Report for the US checkout: status, currency, totals, payment handlers/services available
   to the buyer, fulfillment methods and delivery options with prices, and every
   error/warning message verbatim.
4. If step 2 succeeded, repeat once with a Saudi Arabia address context.
5. Finish with a compact markdown report:
   - funnel verdict: can a buyer technically complete checkout today (payment methods present or not)
   - the exact blockers found (payment handlers missing, delivery rates missing, messages)
   - products/variants used and checkout ids
   Provide evidence from tool responses, not assumptions."""


class AuditInput(pydantic.BaseModel):
    mission: str = "Run the checkout funnel audit now."


@workflows.workflow.define(
    name="astaa-store-audit",
    workflow_display_name="ASTAA Store Checkout Audit",
    workflow_description="Audits the ASTAA store checkout funnel via the astaa_7e0d UCP connector. Never completes a checkout.",
)
@uses_connectors(astaa_connector)
class AstaaAuditWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: AuditInput) -> str:
        agent = Agent(
            name="astaa-audit-agent",
            model="mistral-medium-latest",
            instructions=AUDIT_INSTRUCTIONS,
            connectors=[connector(ASTAA_CONNECTOR_NAME)],
        )
        result = await Runner.run(agent=agent, inputs=input.mission)
        return result.final_output
