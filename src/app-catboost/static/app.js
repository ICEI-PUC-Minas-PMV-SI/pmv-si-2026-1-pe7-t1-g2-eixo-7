"use strict";

const CAMPOS_NUMERICOS = [
  "Credit_Score",
  "loan_amount", "term", "income", "dtir1",
];

// Todos os campos do payload (na ordem não importa para a API).
const CAMPOS_PAYLOAD = CAMPOS_NUMERICOS.concat([
  "loan_limit", "Gender", "approv_in_adv", "loan_type", "loan_purpose",
  "Credit_Worthiness", "open_credit", "business_or_commercial",
  "Neg_ammortization", "interest_only", "lump_sum_payment",
  "construction_type", "occupancy_type", "Secured_by", "total_units",
  "credit_type", "co-applicant_credit_type", "age",
  "submission_of_application", "Region", "Security_Type",
]);

const form = document.getElementById("form-previsao");
const btnEnviar = document.getElementById("btn-enviar");

const modal = document.getElementById("modal");
const modalCorpo = document.getElementById("modal-corpo");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!form.reportValidity()) {
    return;
  }

  const dados = {};
  for (const campo of CAMPOS_PAYLOAD) {
    const el = document.getElementById(campo);
    dados[campo] = CAMPOS_NUMERICOS.includes(campo)
      ? Number(el.value)
      : el.value;
  }

  const limiar = document.getElementById("limiar").value;

  btnEnviar.disabled = true;
  btnEnviar.textContent = "Enviando...";

  try {
    const resposta = await fetch(
      "/prever?limiar=" + encodeURIComponent(limiar),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dados),
      }
    );

    const corpo = await resposta.json();

    if (resposta.ok) {
      mostrarResultado(corpo);
    } else {
      mostrarErro(corpo.detail || "Erro ao consultar a API.");
    }
  } catch (erro) {
    mostrarErro("Não foi possível contatar a API: " + erro.message);
  } finally {
    btnEnviar.disabled = false;
    btnEnviar.textContent = "Enviar";
  }
});

function mostrarResultado(r) {
  const inadimplente = r.classe === 1;
  const pct = (r.probabilidade_inadimplencia * 100).toFixed(2);

  modalCorpo.innerHTML = `
    <p class="resultado-rotulo ${inadimplente ? "inadimplente" : "adimplente"}">
      ${r.rotulo}
    </p>
    <div class="barra">
      <div class="barra-preenchida" style="width: ${pct}%;"></div>
    </div>
    <div class="resultado-linha">
      <span>Probabilidade de inadimplência</span>
      <span>${pct}%</span>
    </div>
    <div class="resultado-linha">
      <span>Classe</span>
      <span>${r.classe} (${inadimplente ? "Inadimplente" : "Adimplente"})</span>
    </div>
    <div class="resultado-linha">
      <span>Limiar usado</span>
      <span>${r.limiar}</span>
    </div>
  `;
  abrirModal();
}

function mostrarErro(mensagem) {
  modalCorpo.innerHTML = `
    <p class="resultado-rotulo erro">Erro</p>
    <p>${mensagem}</p>
  `;
  abrirModal();
}

function abrirModal() {
  modal.hidden = false;
}

function fecharModal() {
  modal.hidden = true;
}

document.getElementById("modal-x").addEventListener("click", fecharModal);
document.getElementById("modal-fechar").addEventListener("click", fecharModal);
modal.addEventListener("click", (event) => {
  if (event.target === modal) {
    fecharModal();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !modal.hidden) {
    fecharModal();
  }
});
