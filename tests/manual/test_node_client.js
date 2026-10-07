import { Client } from './extension/background/gradio.js';
(async () => {
  try {
    const client = await Client.connect("http://127.0.0.1:7860");
    const result = await client.predict("/predict", ["O método secreto que os ricos usam para pagar menos impostos!"]);
    console.log("RESULTADO:", JSON.stringify(result));
  } catch(e) { console.error(e) }
  process.exit(0);
})();
