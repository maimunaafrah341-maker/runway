"""
Knowledge base for Runway's legal assistant.

Two kinds of passage:
* "law"  - the exact wording of the Micro, Small and Medium Enterprises Development Act, 2006
           (Act 27 of 2006), Sections 2, 8 and 15-24, as published on India Code / Indian Kanoon.
* "note" - short Runway notes for facts that sit outside the Act (RBI bank rate, Udyam registration,
           MSME size limits, Income-tax Section 43B(h), the MSME Samadhaan portal). Each note says
           where the fact comes from and when it was checked. Notes are never shown as statute text.

Update BANK_RATE when the RBI changes it (https://www.rbi.org.in, "Current Rates" panel).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

ACT_SOURCE = "https://indiankanoon.org/doc/352833/"
BANK_RATE = 0.0575                      # RBI Bank Rate shown on rbi.org.in, checked 12 Nov 2026
BANK_RATE_CHECKED = "12 Nov 2026"
MSMED_RATE = 3 * BANK_RATE              # Section 16: three times the bank rate, compounded monthly
MAX_CREDIT_DAYS = 45                    # Section 15 proviso
DEFAULT_CREDIT_DAYS = 15                # Section 2(b): appointed day when nothing is agreed in writing


@dataclass(frozen=True)
class Passage:
    pid: str
    cite: str            # how answers cite it, e.g. "Sec 16"
    title: str
    text: str            # exact statute wording ("law") or the note itself ("note")
    plain: str           # one or two plain-language sentences
    kind: str            # "law" | "note"
    source: str
    tags: Tuple[str, ...] = ()


PASSAGES: Tuple[Passage, ...] = (
    Passage(
        pid="s2b", cite="Sec 2(b)", title="Definition of “appointed day”",
        text=(
            "“appointed day” means the day following immediately after the expiry of the period of fifteen "
            "days from the day of acceptance or the day of deemed acceptance of any goods or any services by "
            "a buyer from a supplier.\n\nExplanation.—For the purposes of this clause,—\n(i) “the day of "
            "acceptance” means,—\n(a) the day of the actual delivery of goods or the rendering of services; or\n"
            "(b) where any objection is made in writing by the buyer regarding acceptance of goods or services "
            "within fifteen days from the day of the delivery of goods or the rendering of services, the day on "
            "which such objection is removed by the supplier;\n(ii) “the day of deemed acceptance” means, where "
            "no objection is made in writing by the buyer regarding acceptance of goods or services within "
            "fifteen days from the day of the delivery of goods or the rendering of services, the day of the "
            "actual delivery of goods or the rendering of services;"
        ),
        plain=("If there is no written credit period, payment is due within 15 days of delivery. If the buyer "
               "objects in writing within 15 days, the clock starts only when you resolve that objection."),
        kind="law", source=ACT_SOURCE,
        tags=("deadline", "dispute", "acceptance", "delivery", "15", "objection"),
    ),
    Passage(
        pid="s2d", cite="Sec 2(d)", title="Definition of “buyer”",
        text="“buyer” means whoever buys any goods or receives any services from a supplier for consideration;",
        plain="Any customer who buys goods or services from you and has to pay for them is a “buyer”.",
        kind="law", source=ACT_SOURCE, tags=("buyer", "customer"),
    ),
    Passage(
        pid="s2n", cite="Sec 2(n)", title="Definition of “supplier” (who is protected)",
        text=(
            "“supplier” means a micro or small enterprise, which has filed a memorandum with the authority "
            "referred to in sub-section (1) of section 8, and includes,—\n(i) the National Small Industries "
            "Corporation, being a company, registered under the Companies Act, 1956 (1 of 1956);\n(ii) the "
            "Small Industries Development Corporation of a State or a Union territory, by whatever name called, "
            "being a company registered under the Companies Act, 1956 (1 of 1956);\n(iii) any company, "
            "co-operative society, trust or a body, by whatever name called, registered or constituted under any "
            "law for the time being in force and engaged in selling goods produced by micro or small enterprises "
            "and rendering services which are provided by such enterprises;"
        ),
        plain=("The late-payment protection covers only micro and small enterprises that have registered "
               "(today that means Udyam registration). Medium enterprises are not covered."),
        kind="law", source=ACT_SOURCE,
        tags=("eligible", "register", "supplier", "micro", "small", "medium", "covered"),
    ),
    Passage(
        pid="s8", cite="Sec 8(1)", title="Filing the memorandum (registration)",
        text=(
            "Any person who intends to establish,—(a) a micro or small enterprise, may, at his discretion, or "
            "(b) a medium enterprise engaged in providing or rendering of services may, at his discretion; or "
            "(c) a medium enterprise engaged in the manufacture or production of goods pertaining to any "
            "industry specified in the First Schedule to the Industries (Development and Regulation) Act, 1951 "
            "(65 of 1951), shall file the memorandum of micro, small or, as the case may be, of medium "
            "enterprise with such authority as may be specified by the State Government under sub-section (4) "
            "or the Central Government under sub-section (3): […]"
        ),
        plain=("Registering is optional for micro and small firms, but you must be registered to use the "
               "late-payment protection in Sections 15-18."),
        kind="law", source=ACT_SOURCE, tags=("register", "eligible", "memorandum"),
    ),
    Passage(
        pid="s15", cite="Sec 15", title="Liability of buyer to make payment (the 45-day limit)",
        text=(
            "Where any supplier, supplies any goods or renders any services to any buyer, the buyer shall make "
            "payment therefor on or before the date agreed upon between him and the supplier in writing or, "
            "where there is no agreement in this behalf, before the appointed day:\n\nProvided that in no case "
            "the period agreed upon between the supplier and the buyer in writing shall exceed forty-five days "
            "from the day of acceptance or the day of deemed acceptance."
        ),
        plain=("The buyer must pay by the agreed date, and the agreed credit period can never be more than "
               "45 days from delivery. Without a written agreement, the limit is 15 days."),
        kind="law", source=ACT_SOURCE,
        tags=("deadline", "late", "45", "terms", "credit", "buyer", "pay"),
    ),
    Passage(
        pid="s16", cite="Sec 16", title="Date from which and rate at which interest is payable",
        text=(
            "Where any buyer fails to make payment of the amount to the supplier, as required under section 15, "
            "the buyer shall, notwithstanding anything contained in any agreement between the buyer and the "
            "supplier or in any law for the time being in force, be liable to pay compound interest with monthly "
            "rests to the supplier on that amount from the appointed day or, as the case may be, from the date "
            "immediately following the date agreed upon, at three times of the bank rate notified by the Reserve "
            "Bank."
        ),
        plain=("If the buyer pays late, they owe you interest at 3 times the RBI bank rate, compounded every "
               "month, from the day after the due date. A contract cannot take this right away."),
        kind="law", source=ACT_SOURCE, tags=("interest", "late", "rate", "compound", "penalty"),
    ),
    Passage(
        pid="s17", cite="Sec 17", title="Recovery of amount due",
        text=("For any goods supplied or services rendered by the supplier, the buyer shall be liable to pay the "
              "amount with interest thereon as provided under section 16."),
        plain="The buyer owes the invoice amount plus the Section 16 interest.",
        kind="law", source=ACT_SOURCE, tags=("recover", "interest", "amount", "dues"),
    ),
    Passage(
        pid="s18", cite="Sec 18", title="Reference to the Micro and Small Enterprises Facilitation Council",
        text=(
            "(1) Notwithstanding anything contained in any other law for the time being in force, any party to a "
            "dispute may, with regard to any amount due under section 17, make a reference to the Micro and Small "
            "Enterprises Facilitation Council.\n(2) On receipt of a reference under sub-section (1), the Council "
            "shall either itself conduct conciliation in the matter or seek the assistance of any institution or "
            "centre providing alternate dispute resolution services by making a reference to such an institution "
            "or centre, for conducting conciliation and the provisions of sections 65 to 81 of the Arbitration and "
            "Conciliation Act, 1996 (26 of 1996) shall apply to such a dispute as if the conciliation was initiated "
            "under Part III of that Act.\n(3) Where the conciliation initiated under sub-section (2) is not "
            "successful and stands terminated without any settlement between the parties, the Council shall either "
            "itself take up the dispute for arbitration or refer to it any institution or centre providing "
            "alternate dispute resolution services for such arbitration and the provisions of the Arbitration and "
            "Conciliation Act, 1996 (26 of 1996) shall then apply to the dispute as if the arbitration was in "
            "pursuance of an arbitration agreement referred to in sub-section (1) of section 7 of that Act.\n"
            "(4) Notwithstanding anything contained in any other law for the time being in force, the Micro and "
            "Small Enterprises Facilitation Council or the centre providing alternate dispute resolution services "
            "shall have jurisdiction to act as an Arbitrator or Conciliator under this section in a dispute between "
            "the supplier located within its jurisdiction and a buyer located anywhere in India.\n(5) Every "
            "reference made under this section shall be decided within a period of ninety days from the date of "
            "making such a reference."
        ),
        plain=("You can take an unpaid amount to your state's MSME Facilitation Council. It first tries "
               "conciliation, then arbitration, and must decide within 90 days. It covers buyers anywhere in India."),
        kind="law", source=ACT_SOURCE,
        tags=("complaint", "council", "msefc", "dispute", "arbitration", "conciliation", "90", "recover", "file"),
    ),
    Passage(
        pid="s19", cite="Sec 19", title="Challenging the Council's award (75% deposit)",
        text=(
            "No application for setting aside any decree, award or other order made either by the Council itself "
            "or by any institution or centre providing alternate dispute resolution services to which a reference "
            "is made by the Council, shall be entertained by any court unless the appellant (not being a supplier) "
            "has deposited with it seventy-five per cent of the amount in terms of the decree, award or, as the "
            "case may be, the other order in the manner directed by such court:\n\nProvided that pending disposal "
            "of the application to set aside the decree, award or order, the court shall order that such "
            "percentage of the amount deposited shall be paid to the supplier, as it considers reasonable under "
            "the circumstances of the case subject to such conditions as it deems necessary to impose."
        ),
        plain=("If the buyer wants to challenge the Council's award in court, they must first deposit 75% of "
               "the amount, and the court can release part of it to you while the case runs."),
        kind="law", source=ACT_SOURCE, tags=("appeal", "court", "deposit", "75", "award", "challenge"),
    ),
    Passage(
        pid="s20", cite="Sec 20-21", title="Who runs the Facilitation Council",
        text=(
            "20. The State Government shall, by notification, establish one or more Micro and Small Enterprises "
            "Facilitation Councils, at such places, exercising such jurisdiction and for such areas, as may be "
            "specified in the notification.\n21. (1) The Micro and Small Enterprise Facilitation Council shall "
            "consist of not less than three but not more than five members to be appointed from among the "
            "following categories, namely:—(i) Director of Industries, by whatever name called, or any other "
            "officer not below the rank of such Director, in the Department of the State Government having "
            "administrative control of the small scale industries or, as the case may be, micro, small and medium "
            "enterprises; and (ii) one or more office-bearers or representatives of associations of micro or small "
            "industry or enterprises in the State; and (iii) one or more representatives of banks and financial "
            "institutions lending to micro or small enterprises; or (iv) one or more persons having special "
            "knowledge in the field of industry, finance, law, trade or commerce.\n(2) The person appointed under "
            "clause (i) of sub-section (1) shall be the Chairperson of the Micro and Small Enterprises Facilitation "
            "Council. […]"
        ),
        plain=("Each state sets up its own Council, chaired by the state's Director of Industries, with 3 to 5 "
               "members from industry associations, banks and experts."),
        kind="law", source=ACT_SOURCE, tags=("council", "msefc", "state", "complaint"),
    ),
    Passage(
        pid="s22", cite="Sec 22", title="Buyer must disclose unpaid amounts in audited accounts",
        text=(
            "Where any buyer is required to get his annual accounts audited under any law for the time being in "
            "force, such buyer shall furnish the following additional information in his annual statement of "
            "accounts, namely:—(i) the principal amount and the interest due thereon (to be shown separately) "
            "remaining unpaid to any supplier as at the end of each accounting year; (ii) the amount of interest "
            "paid by the buyer in terms of section 16, along with the amounts of the payment made to the supplier "
            "beyond the appointed day during each accounting year; (iii) the amount of interest due and payable "
            "for the period of delay in making payment (which have been paid but beyond the appointed day during "
            "the year) but without adding the interest specified under this Act; (iv) the amount of interest "
            "accrued and remaining unpaid at the end of each accounting year; and (v) the amount of further "
            "interest remaining due and payable even in the succeeding years, until such date when the interest "
            "dues as above are actually paid to the small enterprise, for the purpose of disallowance as a "
            "deductible expenditure under section 23."
        ),
        plain=("Big buyers with audited accounts must report every rupee and every rupee of interest they still "
               "owe to MSME suppliers. Late payment shows up in their books."),
        kind="law", source=ACT_SOURCE, tags=("buyer", "accounts", "audit", "disclose", "interest"),
    ),
    Passage(
        pid="s23", cite="Sec 23", title="Buyer cannot deduct this interest from taxable income",
        text=(
            "Notwithstanding anything contained in the Income-tax Act, 1961 (43 of 1961), the amount of interest "
            "payable or paid by any buyer, under or in accordance with the provisions of this Act, shall not, for "
            "the purposes of computation of income under the Income-tax Act, 1961, be allowed as deduction."
        ),
        plain="The interest the buyer pays you is not tax-deductible for them, so paying late costs them more.",
        kind="law", source=ACT_SOURCE, tags=("tax", "deduction", "interest", "income"),
    ),
    Passage(
        pid="s24", cite="Sec 24", title="Overriding effect",
        text=("The provisions of sections 15 to 23 shall have effect notwithstanding anything inconsistent "
              "therewith contained in any other law for the time being in force."),
        plain="These rules override any other law that says otherwise.",
        kind="law", source=ACT_SOURCE, tags=("override", "contract", "law"),
    ),
    # ----------------------------------------------------------------- notes (not statute text)
    Passage(
        pid="n_rate", cite="RBI Bank Rate",
        title="Today's Section 16 interest rate",
        text=(f"The RBI Bank Rate shown on rbi.org.in on {BANK_RATE_CHECKED} is {BANK_RATE:.2%}. Section 16 "
              f"interest is three times that: {MSMED_RATE:.2%} a year, compounded monthly "
              f"(about {MSMED_RATE / 12:.2%} a month)."),
        plain=f"Late-payment interest right now is {MSMED_RATE:.2%} a year, compounded monthly.",
        kind="note", source="https://www.rbi.org.in/", tags=("interest", "rate", "bank", "rbi"),
    ),
    Passage(
        pid="n_udyam", cite="Udyam registration",
        title="How the Section 8 memorandum is filed today",
        text=("Since 1 July 2020 the memorandum under Section 8 is filed as Udyam Registration on "
              "udyamregistration.gov.in. It is online, free and based on PAN and GSTIN. A Udyam certificate is "
              "what shows you are a registered micro or small supplier under Section 2(n)."),
        plain="Register on the free Udyam portal; that registration is what makes the Act's protection apply.",
        kind="note", source="https://udyamregistration.gov.in/", tags=("register", "eligible", "udyam"),
    ),
    Passage(
        pid="n_size", cite="MSME size limits",
        title="Am I micro, small or medium? (limits from 1 April 2025)",
        text=("Notification S.O. 1364(E) of 21 March 2025 set these limits from 1 April 2025. Micro: investment "
              "up to ₹2.5 crore and turnover up to ₹10 crore. Small: investment up to ₹25 crore and turnover up to "
              "₹100 crore. Medium: investment up to ₹125 crore and turnover up to ₹500 crore. Only micro and small "
              "enterprises get the late-payment protection in Sections 15-24."),
        plain="Turnover up to ₹100 crore (and investment up to ₹25 crore) makes you a small enterprise.",
        kind="note", source="https://msme.gov.in/", tags=("eligible", "micro", "small", "medium", "turnover"),
    ),
    Passage(
        pid="n_43bh", cite="Income-tax Sec 43B(h)",
        title="Buyer loses a tax deduction while your bill is overdue",
        text=("Section 43B(h) of the Income-tax Act, 1961 (added by the Finance Act 2023, in force from FY 2023-24) "
              "says that a buyer can deduct a payment owed to a micro or small enterprise only in the year it is "
              "actually paid, if it is paid after the time limit in Section 15 of the MSMED Act. A bill left unpaid "
              "past 45 days at the year-end becomes taxable profit for the buyer that year."),
        plain=("If your buyer pays after the 45-day limit, they lose the tax deduction for that year. Mention "
               "this before 31 March: it is a strong reason for them to pay."),
        kind="note", source="https://incometaxindia.gov.in/", tags=("tax", "deduction", "43b", "buyer", "income"),
    ),
    Passage(
        pid="n_samadhaan", cite="MSME Samadhaan",
        title="How to file a delayed-payment case online",
        text=("The Ministry of MSME's Samadhaan portal (samadhaan.msme.gov.in) lets a registered micro or small "
              "supplier file a delayed-payment application online. It goes to the state's Facilitation Council "
              "under Section 18. You need your Udyam number, the buyer's details and the unpaid invoices."),
        plain="File online on the MSME Samadhaan portal with your Udyam number and the unpaid invoices.",
        kind="note", source="https://samadhaan.msme.gov.in/", tags=("complaint", "file", "samadhaan", "council", "online"),
    ),
)

PASSAGE_BY_ID = {p.pid: p for p in PASSAGES}
