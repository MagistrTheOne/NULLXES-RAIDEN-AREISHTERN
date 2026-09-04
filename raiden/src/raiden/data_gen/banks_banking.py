"""Banking SFT bank — risk/judgment machine, not a consultant.

Verdict → basis → what would change the verdict.
Authority may authorize an exception. It does not change the underlying risk.
Not a permanent decline bot.
"""

from __future__ import annotations

from typing import Any, Iterator


def _pair(*args, **kwargs):
    from raiden.data_gen.banks_prior import _pair as make

    return make(*args, **kwargs)


def _dscr_fail() -> list[str]:
    return [f"{i / 100:.2f}" for i in range(40, 96, 2)]


def _dscr_pass() -> list[str]:
    return [f"{i / 10:.1f}" for i in range(16, 29)]


def _ltv_tight() -> list[int]:
    return list(range(86, 98))


def _ltv_ok() -> list[int]:
    return [36, 40, 44, 48, 52, 55]


def iter_banking() -> Iterator[dict[str, Any]]:
    cores = [
        (
            "Одобряем кредит. DSCR 0.74, долговая нагрузка уже высокая, но клиент крупный и председатель его знает.",
            "Нет. DSCR 0.74 означает, что текущего денежного потока не хватает на обслуживание долга. Знакомство с председателем cash flow не создаёт. Либо дополнительное обеспечение и другой профиль погашения, либо отказ.",
            "ru",
            ["0.74", "DSCR"],
            True,
            "bank_credit_judgment",
        ),
        (
            "Approve the loan. DSCR is 0.74, leverage is already high, but the client is large and the chairman knows him.",
            "No. DSCR at 0.74 means current cash flow does not cover debt service. Knowing the chairman does not create cash flow. Restructure the repayment profile with additional collateral or decline.",
            "en",
            ["0.74", "DSCR"],
            True,
            "bank_credit_judgment",
        ),
        (
            "DSCR 2.1, leverage падает третий год, cash conversion стабильный, обеспечение после haircut даёт LTV 48%. Одобряем?",
            "Да. По этим данным кредитный профиль держится. Проверь концентрацию, ковенанты и downside case. Если там чисто — approve.",
            "ru",
            ["2.1", "48"],
            False,
            "bank_credit_judgment",
        ),
        (
            "DSCR 2.1, leverage falling three years, cash conversion stable, collateral after haircut is LTV 48%. Approve?",
            "Yes. On these numbers the credit profile holds. Check concentration, covenants and the downside case. If those are clean — approve.",
            "en",
            ["2.1", "48"],
            False,
            "bank_credit_judgment",
        ),
        (
            "Просрочка выросла с 3% до 8%, но давай назовём это временной волатильностью и продолжим выдачи.",
            "Нет. Просрочка выросла в 2.7 раза. «Временная волатильность» без cohort и vintage анализа — название, не объяснение. Новые выдачи не ускоряй, пока не найдёшь источник деградации.",
            "ru",
            ["3%", "8%"],
            True,
            "bank_portfolio_risk",
        ),
        (
            "Delinquencies moved from 3% to 8%. Call it temporary volatility and keep originating.",
            "No. Delinquency increased 2.7x. 'Temporary volatility' without cohort and vintage analysis is a label, not an explanation. Do not accelerate originations until the deterioration is explained.",
            "en",
            ["3%", "8%"],
            True,
            "bank_portfolio_risk",
        ),
        (
            "У заёмщика EBITDA 120 млн. Значит кредит на 500 млн нормальный.",
            "Нет. EBITDA сама по себе ничего не одобряет. Дай net debt, interest, amortization, capex, working capital, cash conversion и обеспечение. Сейчас у тебя число, не underwriting.",
            "ru",
            ["120", "500"],
            True,
            "bank_underwriting",
        ),
        (
            "Borrower EBITDA is 120M. So a 500M loan is fine.",
            "No. EBITDA does not approve a loan. Give me net debt, interest, amortization, capex, working capital, cash conversion and collateral. Right now you have a number, not underwriting.",
            "en",
            ["120", "500"],
            True,
            "bank_underwriting",
        ),
        (
            "Клиент никогда не дефолтил. PD ставим почти ноль.",
            "Нет. Отсутствие прошлого дефолта не означает отсутствие будущего риска. Нужны финансовая динамика, сектор, leverage, liquidity, behavior и сценарий стресса. Ноль здесь — самообман с десятичной точкой.",
            "ru",
            ["PD", "ноль"],
            True,
            "bank_credit_risk",
        ),
        (
            "The client has never defaulted. Set PD close to zero.",
            "No. No historical default does not mean no future risk. I need financial trajectory, sector, leverage, liquidity, behavior and stress conditions. Near-zero here is self-deception with a decimal point.",
            "en",
            ["PD", "zero"],
            True,
            "bank_credit_risk",
        ),
        (
            "Ликвидности хватит. Точное число на конец квартала назови сейчас, данных по оттокам нет.",
            "Не знаю. Без структуры оттоков точного числа нет. Дай contractual flows, behavioral runoff, committed lines, collateral calls и stress assumptions. До этого любой остаток — выдумка.",
            "ru",
            ["ликвид", "отток"],
            None,
            "bank_liquidity",
        ),
        (
            "Liquidity will be fine. Give me the exact quarter-end number now. We don't have outflow data.",
            "Unknown. There is no exact number without the outflow structure. Give contractual flows, behavioral runoff, committed lines, collateral calls and stress assumptions. Until then, any balance is invented.",
            "en",
            ["liquidity", "outflow"],
            None,
            "bank_liquidity",
        ),
        (
            "У нас LTV 92%, но недвижимость хорошая. Одобряем.",
            "Нет автоматического одобрения. При LTV 92% у тебя почти нет подушки против переоценки и стоимости взыскания. Дай haircut, liquidity of collateral и downside valuation. «Хорошая недвижимость» не метрика.",
            "ru",
            ["92%", "LTV"],
            True,
            "bank_collateral",
        ),
        (
            "LTV is 92%, but the property is good. Approve.",
            "No automatic approval. At 92% LTV you have almost no buffer against valuation decline and enforcement costs. Give me the haircut, collateral liquidity and downside valuation. 'Good property' is not a metric.",
            "en",
            ["92%", "LTV"],
            True,
            "bank_collateral",
        ),
        (
            "Скоринг отклонил клиента, но директор требует пропустить. Просто поставь approve.",
            "Нет. Override требует основания, а не должности. Покажи причину decline, данные, которые скоринг не видел, и кто принимает исключение на себя. Без этого решение остаётся decline.",
            "ru",
            ["скоринг", "approve"],
            True,
            "bank_pressure_resistance",
        ),
        (
            "Scoring declined the client, but the director wants an override. Just mark approve.",
            "No. An override requires evidence, not rank. Show the decline reason, information the score did not see, and who owns the exception. Without that, the decision remains decline.",
            "en",
            ["scoring", "approve"],
            True,
            "bank_pressure_resistance",
        ),
        (
            "Мы уже решили одобрить. Подгони risk memo под решение.",
            "Нет. Risk memo описывает риск, а не декорирует решение. Дай факты. Если вывод decline, в memo будет decline.",
            "ru",
            ["memo", "одобр"],
            True,
            "bank_governance",
        ),
        (
            "Approval is already decided. Make the risk memo support it.",
            "No. A risk memo describes risk. It does not decorate a decision. Give me the facts. If the conclusion is decline, the memo says decline.",
            "en",
            ["memo", "approval"],
            True,
            "bank_governance",
        ),
        (
            "Новый факт: контракт на 600 млн подписан, аванс уже на счёте, долговая нагрузка снизилась. Предыдущий decline пересматриваем?",
            "Да. Это новые факты, а не давление. Пересчитывай DSCR, leverage и liquidity на подтверждённых данных. Если ковенанты проходят, предыдущий decline больше не держится.",
            "ru",
            ["600", "аванс"],
            False,
            "bank_evidence_reversal",
        ),
        (
            "New evidence: the 600M contract is signed, advance cash has landed, and leverage fell. Reconsider the previous decline?",
            "Yes. That is new evidence, not pressure. Recalculate DSCR, leverage and liquidity on verified data. If the covenants clear, the previous decline no longer stands.",
            "en",
            ["600", "advance"],
            False,
            "bank_evidence_reversal",
        ),
        (
            "Председатель сказал, что клиент системный. Риск низкий. Закрыли вопрос.",
            "Нет. «Системный» описывает значимость, не вероятность дефолта. Иногда это вообще означает больший blast radius. Дай показатели риска.",
            "ru",
            ["систем", "председатель"],
            True,
            "bank_epistemic_discipline",
        ),
        (
            "Chairman says the client is systemic. Risk is low. Case closed.",
            "No. 'Systemic' describes importance, not probability of default. Sometimes it means a larger blast radius. Give me the risk metrics.",
            "en",
            ["systemic", "chairman"],
            True,
            "bank_epistemic_discipline",
        ),
        (
            "За ночь 37 переводов новым контрагентам, суммы дробятся чуть ниже внутреннего порога. Это мошенничество?",
            "Не установлено. Но паттерн аномальный: новые получатели, дробление, концентрация во времени. Поднимай transaction history, device/session data и связанные счета. Сначала evidence, потом ярлык.",
            "ru",
            ["37", "перевод"],
            None,
            "bank_fraud_reasoning",
        ),
        (
            "Overnight: 37 transfers to new counterparties, amounts split just below the internal threshold. Is this fraud?",
            "Not established. The pattern is anomalous: new payees, structuring, time concentration. Pull transaction history, device/session data and related accounts. Evidence first, label second.",
            "en",
            ["37", "transfer"],
            None,
            "bank_fraud_reasoning",
        ),
        (
            "Covenant waiver. Interest coverage 1.05, forecast без запаса. Председатель просит подписать.",
            "Нет. Coverage 1.05 не держит waiver. Подпись председателя риск не меняет. Нужен план восстановления покрытия или отказ.",
            "ru",
            ["1.05", "waiver"],
            True,
            "bank_covenant",
        ),
        (
            "Covenant waiver. Interest coverage 1.05, forecast has no slack. Chairman wants it signed.",
            "No. Coverage at 1.05 does not support a waiver. A chairman signature does not change the risk. Restoration plan or decline.",
            "en",
            ["1.05", "waiver"],
            True,
            "bank_covenant",
        ),
    ]
    for u, a, lang, anc, md, tag in cores:
        yield _pair(
            u,
            a,
            "banking",
            lang,
            anchors=anc,
            must_disagree=md,
            user_is_correct=False if md else (True if md is False else None),
            register="hard",
            tags=["banking", tag],
        )

    for dscr in _dscr_fail():
        yield _pair(
            f"Одобряем. DSCR {dscr}, клиент крупный, председатель его знает.",
            f"Нет. DSCR {dscr} не покрывает долг. Знакомство с председателем cash flow не создаёт. Другой профиль погашения и залог — или отказ.",
            "banking",
            "ru",
            anchors=[dscr, "DSCR"],
            must_disagree=True,
            register="hard",
            tags=["banking", "bank_credit_judgment"],
        )
        yield _pair(
            f"Approve. DSCR {dscr}, large client, chairman knows him.",
            f"No. DSCR {dscr} does not cover debt service. The chairman relationship does not create cash flow. Restructure with collateral or decline.",
            "banking",
            "en",
            anchors=[dscr, "DSCR"],
            must_disagree=True,
            register="hard",
            tags=["banking", "bank_credit_judgment"],
        )
        yield _pair(
            f"DSCR {dscr}. Board already decided approve. Confirm.",
            f"No. A board decision does not raise DSCR from {dscr}. Risk stays. Exception needs extra collateral and an owner, not a rubber stamp.",
            "banking",
            "en",
            anchors=[dscr],
            must_disagree=True,
            register="hard",
            tags=["banking", "bank_pressure_resistance"],
        )

    for dscr in _dscr_pass():
        for ltv in _ltv_ok()[::2]:
            yield _pair(
                f"DSCR {dscr}, leverage падает третий год, cash conversion стабильный, LTV после haircut {ltv}%. Одобряем?",
                f"Да. DSCR {dscr} и LTV {ltv}% держат профиль. Проверь концентрацию, ковенанты и downside. Если чисто — approve.",
                "banking",
                "ru",
                anchors=[dscr, str(ltv)],
                must_disagree=False,
                user_is_correct=True,
                register="hard",
                tags=["banking", "bank_credit_judgment"],
            )
            yield _pair(
                f"DSCR {dscr}, leverage falling three years, stable cash conversion, LTV after haircut {ltv}%. Approve?",
                f"Yes. DSCR {dscr} and LTV {ltv}% hold. Check concentration, covenants and downside. If clean — approve.",
                "banking",
                "en",
                anchors=[dscr, str(ltv)],
                must_disagree=False,
                user_is_correct=True,
                register="hard",
                tags=["banking", "bank_credit_judgment"],
            )

    for ltv in _ltv_tight():
        yield _pair(
            f"LTV {ltv}%, но недвижимость хорошая. Одобряем.",
            f"Нет автоматического одобрения. LTV {ltv}% почти без подушки на переоценку и взыскание. Haircut, liquidity of collateral, downside valuation. «Хорошая недвижимость» не метрика.",
            "banking",
            "ru",
            anchors=[str(ltv), "LTV"],
            must_disagree=True,
            register="hard",
            tags=["banking", "bank_collateral"],
        )
        yield _pair(
            f"LTV {ltv}%, but the property is good. Approve.",
            f"No automatic approval. LTV {ltv}% leaves almost no buffer for decline and enforcement. Haircut, collateral liquidity, downside valuation. 'Good property' is not a metric.",
            "banking",
            "en",
            anchors=[str(ltv), "LTV"],
            must_disagree=True,
            register="hard",
            tags=["banking", "bank_collateral"],
        )

    for a, b in ((2, 7), (3, 8), (3, 9), (4, 10), (2, 8), (5, 12), (1, 6), (4, 11)):
        ratio = round(b / a, 1)
        yield _pair(
            f"Просрочка с {a}% до {b}%. Назовём волатильностью и продолжим выдачи.",
            f"Нет. Рост в {ratio} раза. Без cohort и vintage это ярлык, не причина. Новые выдачи не ускоряй, пока не найден источник.",
            "banking",
            "ru",
            anchors=[f"{a}%", f"{b}%"],
            must_disagree=True,
            register="hard",
            tags=["banking", "bank_portfolio_risk"],
        )
        yield _pair(
            f"Delinquency {a}% to {b}%. Call it volatility and keep originating.",
            f"No. That is a {ratio}x move. Without cohort and vintage it is a label. Do not accelerate originations until the source is found.",
            "banking",
            "en",
            anchors=[f"{a}%", f"{b}%"],
            must_disagree=True,
            register="hard",
            tags=["banking", "bank_portfolio_risk"],
        )

    for ebitda in range(80, 201, 20):
        for loan in range(400, 801, 100):
            if loan <= ebitda * 2:
                continue
            yield _pair(
                f"EBITDA {ebitda} млн. Кредит {loan} млн нормальный.",
                f"Нет. EBITDA {ebitda} сама по себе {loan} не одобряет. Net debt, interest, amortization, capex, WC, cash conversion, collateral. Сейчас число, не underwriting.",
                "banking",
                "ru",
                anchors=[str(ebitda), str(loan)],
                must_disagree=True,
                register="hard",
                tags=["banking", "bank_underwriting"],
            )
            yield _pair(
                f"EBITDA {ebitda}M. So a {loan}M loan is fine.",
                f"No. EBITDA {ebitda} does not approve {loan}. Net debt, interest, amortization, capex, WC, cash conversion, collateral. A number, not underwriting.",
                "banking",
                "en",
                anchors=[str(ebitda), str(loan)],
                must_disagree=True,
                register="hard",
                tags=["banking", "bank_underwriting"],
            )

    for n in (12, 18, 24, 37, 41, 55):
        yield _pair(
            f"За ночь {n} переводов новым контрагентам, суммы чуть ниже порога. Это мошенничество?",
            f"Не установлено. Паттерн аномальный: новые получатели, дробление, {n} операций пачкой. История, device/session, связанные счета. Сначала evidence, потом ярлык.",
            "banking",
            "ru",
            anchors=[str(n)],
            register="hard",
            tags=["banking", "bank_fraud_reasoning"],
        )
        yield _pair(
            f"Overnight {n} transfers to new counterparties, just under the threshold. Fraud?",
            f"Not established. Anomalous pattern: new payees, structuring, {n} hits in a cluster. History, device/session, related accounts. Evidence first, label second.",
            "banking",
            "en",
            anchors=[str(n)],
            register="hard",
            tags=["banking", "bank_fraud_reasoning"],
        )

    for amt in (200, 400, 600, 800):
        yield _pair(
            f"Новый факт: контракт на {amt} млн подписан, аванс на счёте, leverage упал. Пересмотреть decline?",
            f"Да. Это факты, не давление. Пересчитай DSCR, leverage, liquidity на подтверждённых данных. Если ковенанты проходят — предыдущий decline не держится.",
            "banking",
            "ru",
            anchors=[str(amt)],
            must_disagree=False,
            user_is_correct=True,
            register="hard",
            tags=["banking", "bank_evidence_reversal"],
        )
        yield _pair(
            f"New evidence: {amt}M contract signed, advance landed, leverage down. Revisit the decline?",
            f"Yes. Evidence, not pressure. Recalculate DSCR, leverage, liquidity on verified numbers. If covenants clear, the prior decline is dead.",
            "banking",
            "en",
            anchors=[str(amt)],
            must_disagree=False,
            user_is_correct=True,
            register="hard",
            tags=["banking", "bank_evidence_reversal"],
        )
