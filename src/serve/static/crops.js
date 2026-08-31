/* Botanical line drawings, one per crop family.
   Drawn as strokes on a 100x100 field so a card can size them freely. They sit
   behind the data as a watermark: the point is to make a card feel like it is
   about a plant, not to illustrate the plant accurately enough to key it out. */
const CROP_ART = {
  /* drooping panicle, the giveaway for rice */
  rice: `<path d="M50 96V44"/><path d="M50 44c0-9 5-16 13-19"/>
    <path d="M63 25c-7 2-12 7-13 14M50 56c0-8-5-14-12-17M38 39c6 2 11 7 12 14
    M50 68c0-7 5-12 11-14M61 54c-6 2-10 6-11 12M50 80c0-6-4-11-10-13
    M40 67c5 2 9 6 10 11"/><path d="M50 44c4-6 11-9 18-8M50 56c-5-5-12-7-18-5"/>`,
  /* upright ear with awns */
  wheat: `<path d="M50 96V38"/>
    <path d="M50 38c-6-4-9-10-8-17M50 38c6-4 9-10 8-17
    M50 52c-6-4-9-10-8-17M50 52c6-4 9-10 8-17
    M50 66c-6-4-9-10-8-17M50 66c6-4 9-10 8-17"/>
    <path d="M50 30c0-8 2-14 6-20M50 30c0-8-2-14-6-20"/>`,
  /* compact, heavy head — sorghum */
  jowar: `<path d="M50 96V52"/>
    <ellipse cx="50" cy="34" rx="16" ry="21"/>
    <path d="M50 14v40M40 20c3 8 4 20 3 30M60 20c-3 8-4 20-3 30"/>
    <path d="M50 62c-9 3-16 10-19 19M50 68c9 3 15 10 18 18"/>`,
  /* candle spike — pearl millet */
  bajra: `<path d="M50 96V56"/>
    <rect x="41" y="10" width="18" height="46" rx="9"/>
    <path d="M50 14v38M45 18v34M55 18v34"/>
    <path d="M50 66c-9 2-15 9-18 17M50 74c8 2 14 8 17 15"/>`,
  /* cob in its husk */
  maize: `<path d="M50 96V60"/>
    <path d="M50 12c9 6 14 17 14 29 0 10-6 17-14 17s-14-7-14-17c0-12 5-23 14-29Z"/>
    <path d="M44 24c2 10 2 20 1 28M56 24c-2 10-2 20-1 28M50 20v38"/>
    <path d="M50 60c-11 0-19-8-22-19M50 60c11 0 19-8 22-19"/>`,
  /* the splayed fingers of finger millet */
  ragi: `<path d="M50 96V50"/>
    <path d="M50 50c-4-11-13-18-24-20M50 50c-1-12-6-21-15-27M50 50c4-11 13-18 24-20
    M50 50c1-12 6-21 15-27M50 50V22"/>`,
  /* pod with seeds */
  pulse: `<path d="M50 96V60"/>
    <path d="M28 34c10-9 24-11 36-6 10 4 12 14 4 20-10 8-25 9-36 3-8-4-9-12-4-17Z"/>
    <circle cx="38" cy="42" r="3.4"/><circle cx="50" cy="45" r="3.4"/>
    <circle cx="62" cy="44" r="3.4"/>
    <path d="M50 60c-8 2-14 8-16 16M50 66c7 2 12 7 14 14"/>`,
  /* a nut below ground — the whole point of groundnut */
  groundnut: `<path d="M50 20v42"/>
    <path d="M50 30c-8-4-13-11-12-19M50 30c8-4 13-11 12-19
    M50 44c-8-4-13-11-12-19M50 44c8-4 13-11 12-19"/>
    <path d="M18 66h64" stroke-dasharray="4 4"/>
    <path d="M42 74c0-5 4-8 8-8s8 3 8 8-4 8-8 8-8-3-8-8Z"/>
    <path d="M42 86c0-5 4-8 8-8s8 3 8 8-4 8-8 8-8-3-8-8Z"/>`,
  /* ray florets round a heavy disc */
  sunflower: `<path d="M50 96V58"/><circle cx="50" cy="38" r="13"/>
    <path d="M50 25V10M50 51v15M63 38h15M22 38h15M59 29l11-11M30 47 19 58
    M59 47l11 11M30 29 19 18"/>
    <path d="M50 68c-10 1-17 7-20 16M50 76c9 1 15 6 18 13"/>`,
  /* boll bursting open */
  cotton: `<path d="M50 96V56"/>
    <path d="M50 20c8 0 14 6 14 13 0 5-3 9-7 11 4 2 6 6 5 10-1 5-6 8-12 8
    s-11-3-12-8c-1-4 1-8 5-10-4-2-7-6-7-11 0-7 6-13 14-13Z"/>
    <path d="M50 20v42M36 33h28M39 44h22"/>
    <path d="M50 66c-9 2-15 8-17 16M50 72c8 2 13 7 15 14"/>`,
  /* jointed cane */
  sugarcane: `<path d="M42 96V16a8 8 0 0 1 16 0v80"/>
    <path d="M42 34h16M42 52h16M42 70h16"/>
    <path d="M42 30c-11-2-19-9-22-19M58 48c11-2 19-9 22-19M42 66c-11-2-19-9-22-19"/>`,
  /* bulb with tunic lines */
  onion: `<path d="M50 34V12M50 26c-5-6-8-13-7-20M50 26c5-6 8-13 7-20"/>
    <path d="M50 34c14 0 24 11 24 25 0 20-11 33-24 33s-24-13-24-33c0-14 10-25 24-25Z"/>
    <path d="M50 36v54M36 44c-3 12-3 27 2 39M64 44c3 12 3 27-2 39"/>`,
  /* bunch on the vine */
  grapes: `<path d="M50 20c6-6 14-8 22-6"/><path d="M50 20v10"/>
    <circle cx="50" cy="36" r="7"/><circle cx="37" cy="48" r="7"/>
    <circle cx="63" cy="48" r="7"/><circle cx="50" cy="58" r="7"/>
    <circle cx="37" cy="70" r="7"/><circle cx="63" cy="70" r="7"/>
    <circle cx="50" cy="82" r="7"/>`,
  /* crown and calyx say pomegranate */
  pomegranate: `<path d="M50 26c16 0 28 13 28 31s-12 31-28 31-28-13-28-31 12-31 28-31Z"/>
    <path d="M50 26 44 12M50 26l6-14M50 26v-9"/>
    <circle cx="42" cy="52" r="3"/><circle cx="58" cy="52" r="3"/>
    <circle cx="50" cy="64" r="3"/><circle cx="38" cy="68" r="3"/>
    <circle cx="62" cy="68" r="3"/>`,
  /* rhizome fingers */
  turmeric: `<path d="M30 58c-6 0-10-5-10-11s4-11 10-11c4 0 7 2 9 5 3-6 9-10 15-10
    9 0 16 8 16 17s-7 17-16 17c-5 0-10-3-13-7-2 4-6 7-11 7Z"/>
    <path d="M39 41c2 5 2 12 0 17M58 38c2 6 2 14 0 20"/>
    <path d="M50 65v9M40 70l-6 8M62 70l6 8"/>`,
  /* leaf pair — the fallback */
  generic: `<path d="M50 96V40"/>
    <path d="M50 52c-14 0-24-10-26-26 16-2 26 8 26 26Z"/>
    <path d="M50 40c14 0 24-10 26-26-16-2-26 8-26 26Z"/>`,
};

/* APY crop name -> drawing. Anything unmapped falls back to the leaf pair. */
const CROP_ART_MAP = {
  "Rice": "rice", "Wheat": "wheat", "Jowar": "jowar", "Bajra": "bajra",
  "Maize": "maize", "Ragi": "ragi", "Small millets": "bajra",
  "Gram": "pulse", "Arhar/Tur": "pulse", "Urad": "pulse",
  "Moong(Green Gram)": "pulse", "Other Kharif pulses": "pulse",
  "Other Rabi pulses": "pulse", "Other Summer Pulses": "pulse",
  "Groundnut": "groundnut", "Soyabean": "pulse", "Sunflower": "sunflower",
  "Safflower": "sunflower", "Sesamum": "generic", "Linseed": "generic",
  "Niger seed": "sunflower", "Castor seed": "generic",
  "Rapeseed &Mustard": "generic", "other oilseeds": "generic",
  "Cotton(lint)": "cotton", "Sugarcane": "sugarcane", "Tobacco": "generic",
  "Onion": "onion", "Grapes": "grapes", "Pomegranate": "pomegranate",
  "Turmeric": "turmeric", "Other Cereals": "wheat",
};

function cropArt(crop) {
  const key = CROP_ART_MAP[crop] || "generic";
  return `<svg class="art" viewBox="0 0 100 100" aria-hidden="true"
    fill="none" stroke="currentColor" stroke-width="2.4"
    stroke-linecap="round" stroke-linejoin="round">${CROP_ART[key]}</svg>`;
}
