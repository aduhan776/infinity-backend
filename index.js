import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenerativeAI } from '@google/generative-ai';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

function makeGenerativePart(base64DataUrl) {
  const match = base64DataUrl.match(/^data:(.*);base64,(.*)$/);
  if (!match) return null;
  return {
    inlineData: {
      data: match[2],
      mimeType: match[1]
    },
  };
}

app.get('/', (req, res) => {
  res.send('Infinity Bulletproof Engine with Auto-Retry Layer Live! ⚡🚀');
});

// ==========================================
// 🎯 ROUTE 1: CLEAN & HIGH-VARIETY TEST GENERATION ENGINE
// ==========================================
app.post('/api/generate-test', async (req, res) => {
  try {
    const { exam, topic, count, type, difficulty, language } = req.body;

    if (!topic) return res.status(400).json({ error: "Topic missing bhai!" });

    const totalRequested = parseInt(count) || 5;
    const targetExam = exam || "Competitive Exam";
    const qType = type || "Objective";
    const diffLevel = difficulty || "Medium";
    const lang = language || "English";

    const model = genAI.getGenerativeModel({ 
      model: "gemini-3.1-flash-lite",
      generationConfig: { responseMimeType: "application/json" }
    });

    const MAX_CHUNK_SIZE = 15;
    let allCompiledQuestions = [];
    let remainingQuestions = totalRequested;

    console.log(`🚀 Starting Safe Pipeline: Requested ${totalRequested} Qs for topic "${topic}"`);

    while (remainingQuestions > 0) {
      const currentChunkSize = Math.min(MAX_CHUNK_SIZE, remainingQuestions);
      
      // Ek chota sa random seed peeche lga rhenge taaki AI ka mood har request mein fresh rahe
      const sessionSeed = Math.random().toString(36).substring(7);
      let prompt = "";

      if (qType === 'Objective') {
        prompt = `Gen ${currentChunkSize} MCQs for ${targetExam}. Topic: "${topic}". Lang: ${lang}. Diff: ${diffLevel}. Seed: ${sessionSeed}
        [DEPTH]: Comprehensively cover the entire sub-topics, basic concepts, theorems, equations, and advanced numerical/logical variants of "${topic}". No repetitions.
        [TYPOGRAPHY]: DO NOT use LaTeX ($ or \\) or raw keyboard laziness (^, _, *, brackets for limits). Use human-readable text with native UNICODE characters. Write actual superscripts (A³, x²), subscripts (λ₁, x₂), Greek letters (λ, θ, π, α, β), and signs (∫, ∂, ±, ∞). Present integrals cleanly as plain text: "Evaluate the integral of [function] from [limit a] to [limit b]".
        [STRUCTURE]: For multi-statement or list-based questions, do NOT lump text into a single paragraph. Use explicit dual newlines (\\n\\n) to break lines, numbered points, or listed conditions cleanly so they render on separate lines.
        [OPTIONS]: Distribute 'correctOptionIndex' randomly across 0,1,2,3 to eliminate bias.
        JSON schema: {"questions": [{"id":0,"question":"","options":["","","",""],"correctOptionIndex":0,"explanation":""}]}.
        Explanation: Max 20 words direct core fact.`;
      } else {
        prompt = `Gen ${currentChunkSize} descriptive questions for ${targetExam}. Topic: "${topic}". Lang: ${lang}. Diff: ${diffLevel}. Seed: ${sessionSeed}
        [DEPTH]: Target thorough syllabus mapping of "${topic}" matching a real professional descriptive examination challenge sheet.
        [TYPOGRAPHY]: DO NOT use LaTeX ($ or \\) or lazy typing symbols (^, _, *). Use pure text containing native browser-supported UNICODE formatting for exponents (x³, y²), subscripts (C₁, C₂), and operators (∫, ∂, ±, ∞, λ, θ, π). Integrals must look like: "Find the integral of [function] within boundaries [a] and [b]".
        [STRUCTURE]: Break long scenarios, context data, or multiple directives using clear escaped dual line-breaks (\\n\\n) instead of clumping.
        JSON schema: {"questions": [{"id":0,"question":"","explanation":""}]}.
        Explanation: Max 35 words strict core valuation framework points.`;
      }

      let retries = 3;
      let responseText = "";
      
      while (retries > 0) {
        try {
          console.log(`📦 Requesting batch of ${currentChunkSize} questions... (Attempts left: ${retries})`);
          const result = await model.generateContent(prompt);
          const response = await result.response;
          responseText = response.text();
          break; 
        } catch (apiError) {
          retries--;
          console.warn(`⚠️ Google Server high demand spike hit. Retrying in 2 seconds...`);
          if (retries === 0) throw apiError; 
          await new Promise(resolve => setTimeout(resolve, 2000)); 
        }
      }

      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error("Invalid AI JSON stream layer returned.");

      const parsedData = JSON.parse(jsonMatch[0]);
      if (parsedData.questions && Array.isArray(parsedData.questions)) {
        allCompiledQuestions = [...allCompiledQuestions, ...parsedData.questions];
      }

      remainingQuestions -= currentChunkSize;
    }

    const finalIndexedQuestions = allCompiledQuestions.map((q, index) => ({
      ...q,
      id: index
    }));

    console.log(`✅ Pipeline Success! Successfully merged ${finalIndexedQuestions.length} targeted questions.`);

    res.json({
      success: true,
      topic: topic,
      questions: finalIndexedQuestions
    });

  } catch (error) {
    console.error("❌ Final Backend Error Log:", error);
    res.status(500).json({ 
      success: false, 
      error: error.message || "Unknown locha occurred in pipeline module."
    });
  }
});

// ======================================================================
// 📝 ROUTE 2: MULTIMODAL SUBJECTIVE EVALUATION GATEWAY
// ======================================================================
app.post('/api/evaluate-subjective', async (req, res) => {
  try {
    const { question, userAnswer, uploadedFiles, testTitle, maxMarks } = req.body;

    if (!question) {
      return res.status(400).json({ success: false, error: "Question metadata reference is missing!" });
    }

    const evaluationModel = genAI.getGenerativeModel({
      model: "gemini-3.1-flash-lite",
      generationConfig: { responseMimeType: "application/json" }
    });

    const parsedMaxMarks = parseFloat(maxMarks) || 10.0;
    const computedTestTitle = testTitle || "Descriptive Assessment Challenge";

    const evaluationPrompt = `
      You are an elite, highly critical senior examiner executing rigorous assessments for Project Infinity.
      Your goal is to inspect the student's handwritten answer sheets (provided as images) and supplementary text notes against the given question text.
      
      [CRITICAL CONTEXT ANALYSIS MATRIX]
      - Target Test Context Name: "${computedTestTitle}"
      - Target Question Statement: "${question}"
      - Maximum Possible Marks Allotted: ${parsedMaxMarks}
      - Student Supplementary Text Input: "${userAnswer || "None provided"}"

      [EVALUATION RULES & STANDARDS GATEWAY]
      1. DYNAMIC CONTEXT ADAPTATION: You must adapt your grading severity instantly to the exam level implied by the test title:
         - Civil Services (e.g., UPSC, State PSC): Look for multi-dimensional analysis, administrative alignment, and logical flow. Highly academic grading.
         - Secondary School Boards (e.g., CBSE, ICSE, Class 10/12): Look strictly for exact textual definitions, crucial key terms, and textbook points compliance.
         - Other Exams (e.g., SSC, Descriptive Banking, Technical): Prioritize precise factual accuracy, structural formatting, and to-the-point answers.
      2. EXTREME STRICTNESS MODE: Do not award marks casually. Be exceptionally stringent. Deduct fractional points for poor structuring, vague concepts, or missing references. 
      3. SCALE-PROPORTIONAL SCORING: Provide a strict numeric 'score_given' that scales precisely between 0 and ${parsedMaxMarks}.
      4. COMPACT NO-FLUFF OUTPUT: Do not give general commentary or essays. Provide a concise points summary and crisp constructive feedback matching the target schema.

      [OUTPUT EXPECTED SCHEMA MAPPING]
      Return exclusively a JSON object matching this exact architectural structure:
      {
        "score_given": 0.0,
        "ai_evaluation": {
          "student_points": [
            "Point 1 summarizing accurately a concept the student managed to cover based on the image text analysis.",
            "Point 2 highlighting another specific key provision or keyword noted in the handwritten draft."
          ],
          "scope_of_improvement": "A mid-length concise review detailing exactly what critical criteria was missing or how to format this answer better to secure full marks."
        }
      }
    `;

    const generativePayloadParts = [evaluationPrompt];
    
    if (uploadedFiles && Array.isArray(uploadedFiles)) {
      uploadedFiles.forEach(file => {
        if (file.url && file.url.startsWith('data:')) {
          const mappedPart = makeGenerativePart(file.url);
          if (mappedPart) generativePayloadParts.push(mappedPart);
        }
      });
    }

    let retries = 3;
    let evaluationResultText = "";

    while (retries > 0) {
      try {
        console.log(`🔍 Initializing Extreme Strict Evaluation via Gemini Multimodal Vision Layer...`);
        const result = await evaluationModel.generateContent(generativePayloadParts);
        const finalResponse = await result.response;
        evaluationResultText = finalResponse.text();
        break;
      } catch (err) {
        retries--;
        console.warn(`⚠️ High demand server spike hit subjective evaluator. Retrying batch pipeline...`);
        if (retries === 0) throw err;
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }

    const cleanJsonMatch = evaluationResultText.match(/\{[\s\S]*\}/);
    if (!cleanJsonMatch) throw new Error("Evaluation Engine failed to output a reliable structured matrix response.");

    const finalEvaluatedPayload = JSON.parse(cleanJsonMatch[0]);
    console.log(`✅ Subjective Evaluation Complete! Core Score Compiled: ${finalEvaluatedPayload.score_given}/${parsedMaxMarks}`);

    res.json({
      success: true,
      evaluation: finalEvaluatedPayload
    });

  } catch (error) {
    console.error("❌ Subjective Evaluator Pipeline Error:", error);
    res.status(500).json({
      success: false,
      error: error.message || "An error locked up the subjective processing module matrix."
    });
  }
});

app.listen(PORT, () => console.log(`🔥 Server active on port: ${PORT}`));