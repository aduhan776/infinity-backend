import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenerativeAI } from '@google/generative-ai';

dotenv.config();

// 🚨 1. FAIL-FAST STARTUP LAYER
if (!process.env.GEMINI_API_KEY) {
  console.error("❌ CRITICAL BOOT FAILURE: GEMINI_API_KEY environment variable is missing inside backend .env file!");
  process.exit(1);
}

const app = express();
const PORT = process.env.PORT || 5000;

// 🔒 2. HARDENED CORS SECURITY POLICY
const allowedOrigins = process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',') : ['http://localhost:5173'];
app.use(cors({
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Access blocked by Infinity Security Framework Gateway (CORS Violation).'));
    }
  }
}));

// Reduced payload window safely from 50mb to 15mb to prevent malicious DoS memory utilization overheads
app.use(express.json({ limit: '15mb' }));
app.use(express.urlencoded({ limit: '15mb', extended: true }));

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
  res.send('Infinity Airtight Production Engine Live! ⚡🚀');
});

// ======================================================================
// 🎯 ROUTE 1: CLEAN & HIGH-VARIETY TEST GENERATION ENGINE (Hardened)
// ======================================================================
app.post('/api/generate-test', async (req, res) => {
  try {
    const { exam, subject, topic, count, type, difficulty, language } = req.body;

    if (!subject) return res.status(400).json({ error: "Subject / Section missing bhai!" });

    // 🚨 3. SERVER SIDE QUESTION COUNT BOUNDS CONSTRAINT
    const rawRequested = parseInt(count) || 5;
    const totalRequested = Math.min(50, Math.max(3, rawRequested)); 

    const targetExam = exam || "Competitive Exam";
    const targetSubject = subject;
    const topicFocusPhrase = topic && topic.trim()
      ? `Specific Topic Focus: "${topic}".`
      : `No specific narrow topic given — cover general questions broadly across this subject/section.`;
    const qType = type || "Objective";
    const diffLevel = difficulty || "Medium";
    const lang = language || "English";

    const model = genAI.getGenerativeModel({ 
      model: "gemini-3.1-flash-lite",
      // 🎲 temperature raised from the model's default toward its upper range to
      // meaningfully increase question variety between separate generation calls
      // on the same topic — this alone won't guarantee zero repeats (that needs
      // an actual exclusion-list mechanism), but reduces how often the model
      // clusters around the same "most likely" questions for a narrow topic.
      generationConfig: { responseMimeType: "application/json", temperature: 1.3 }
    });

    const MAX_CHUNK_SIZE = 15;
    let allCompiledQuestions = [];
    let remainingQuestions = totalRequested;

    console.log(`🚀 Pipeline Active: Processing requested ${totalRequested} Qs safely for subject "${targetSubject}"${topic ? ` (topic: "${topic}")` : ''}`);

    while (remainingQuestions > 0) {
      const currentChunkSize = Math.min(MAX_CHUNK_SIZE, remainingQuestions);
      const sessionSeed = Math.random().toString(36).substring(7);
      let prompt = "";

      if (qType === 'Objective') {
        prompt = `Generate EXACTLY ${currentChunkSize} unique Multiple Choice Questions (MCQs) for ${targetExam}. Subject/Section: "${targetSubject}". ${topicFocusPhrase} Lang: ${lang}. Seed: ${sessionSeed}
        
        [STRICT COUNT CONSTRAINT]: Your JSON response array MUST contain exactly ${currentChunkSize} question objects inside the "questions" array. Absolutely do not generate more than or less than ${currentChunkSize} questions.
        
        [TARGET EXAM & TOPIC ADAPTATION]: CRITICAL RULE! Align questions precisely with the syllabus profile of ${targetExam}. 
        - For Humanities, General Studies, and Conceptual topics (e.g., Geography, History, Indian Polity, General Economics, Ecology): Focus 100% on conceptual clarity, physical mechanisms, features, analytical relationships, and statements. ABSOLUTELY DO NOT invent or force complex mathematical formulas, derivatives, fluid mechanics calculations, or quantitative equations into these questions. Keep it purely aligned to standard GS papers.
        - For Naturally Technical/Quantitative topics (e.g., Pure Mathematics, Physics numericals, Quantitative Chemistry): You are expected to include appropriate formula applications and multi-layered calculations.
        
        [LATEX FORMATTING]: ONLY if mathematical expressions, core variables, formulas, subscripts (e.g., $\\lambda_1$), or superscripts (e.g., $x^2$) are naturally and legitimately required for the topic, wrap them strictly inside inline LaTeX using single dollar signs ($...$). Do NOT artificially force math symbols or LaTeX notation into strictly conceptual humanities/geography text.
        
        [DIFFICULTY CALIBRATION]: Strict Enforcement for "${diffLevel}" level. If difficulty is "Medium", it must strictly match the actual standard core papers of ${targetExam}—make it highly conceptual, analytical, and tricky (ABSOLUTELY NO basic or direct textbook questions). If difficulty is "Tough", make it brutally advanced, elite level, requiring complex structural logic.
        
        [STRUCTURE]: For multi-statement, matching, or list-based questions, do NOT lump statements into one paragraph. You MUST format statements as a clean numbered vertical list (e.g., "Consider the following statements:\\n\\n1. [Statement 1]\\n\\n2. [Statement 2]") with explicit double escaped newlines (\\n\\n) after each item so the frontend renders them beautifully.
        
        [OPTIONS]: Distribute 'correctOptionIndex' randomly across 0,1,2,3.
        JSON schema: {"questions": [{"id":0,"question":"","options":["","","",""],"correctOptionIndex":0,"explanation":""}]}.
        Explanation: Max 20 words core fact wrapped in LaTeX where needed.`;
      } else {
        prompt = `Generate EXACTLY ${currentChunkSize} distinct descriptive/subjective questions for ${targetExam}. Subject/Section: "${targetSubject}". ${topicFocusPhrase} Lang: ${lang}. Seed: ${sessionSeed}
        
        [STRICT COUNT CONSTRAINT]: Your JSON response array MUST contain exactly ${currentChunkSize} question object(s) inside the "questions" array. If the requested count is 3, generate exactly 3 questions. Strict compliance is mandatory.
        
        [TARGET EXAM & TOPIC ADAPTATION]: CRITICAL RULE! Align the descriptive question precisely with the requirements of ${targetExam} mains/written papers.
        - For General Studies/Humanities topics (like Geography, Polity, History, etc.): Ask for analytical evaluation, critical discussions, administrative impacts, or geographical causes. ABSOLUTELY DO NOT force mathematical equations, formulas, or mechanical/computational problems into conceptual topics.
        - For Technical papers (like Physics, Mathematics): Focus on derivations and core quantitative problems.
        
        [LATEX FORMATTING]: ONLY if scientific/mathematical formulas, variables, bounds, or indices are naturally present, wrap them strictly inside inline LaTeX using single dollar signs ($...$). Do NOT invent math formulas for non-mathematical conceptual topics.
        
        [DIFFICULTY CALIBRATION]: Strict Enforcement for "${diffLevel}" level. If difficulty is "Medium", make it deeply conceptual and matching real exam standards. If "Tough", make it highly complex and multi-layered.
        
        [STRUCTURE]: Use explicit double newlines (\\n\\n) to break long problem scenarios, statements, or multi-part directives cleanly into vertical lists or separate paragraphs instead of clumping.
        JSON schema: {"questions": [{"id":0,"question":"","explanation":""}]}.
        Explanation: Max 35 words core grading framework points.`;
      }

      // 🚨 4. UNIFIED RETRY LOOP: retries on BOTH API failures AND JSON parse failures.
      // Gemini is already in JSON mode (responseMimeType: "application/json"), so its raw
      // output is valid JSON — no manual "repair" scanner is needed or safe to use, since
      // such scanners can't reliably tell a real JSON escape (\n, \t, \r) apart from a
      // LaTeX command that happens to start with the same letter (\nu, \tau, \rho, etc.).
      // If parsing genuinely fails, we just ask Gemini again rather than trying to patch it.
      let retries = 3;
      let parsedData = null;

      while (retries > 0) {
        let responseText = "";
        try {
          const result = await model.generateContent(prompt);
          const response = await result.response;
          responseText = response.text();

          const startBrace = responseText.indexOf('{');
          const endBrace = responseText.lastIndexOf('}');
          if (startBrace === -1 || endBrace === -1) {
            throw new Error("Invalid structured AI text response mapping stream.");
          }
          const rawJsonText = responseText.substring(startBrace, endBrace + 1);

          parsedData = JSON.parse(rawJsonText);
          break; // success — exit retry loop
        } catch (err) {
          retries--;
          console.warn(`⚠️ Generation/parse issue hit generator node (${err.message}). Retrying...`);
          if (retries === 0) throw err;
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      }

      if (parsedData && parsedData.questions && Array.isArray(parsedData.questions)) {
        allCompiledQuestions = [...allCompiledQuestions, ...parsedData.questions];
      }

      remainingQuestions -= currentChunkSize;
    }

    const finalIndexedQuestions = allCompiledQuestions.slice(0, totalRequested).map((q, index) => ({
      ...q,
      id: index
    }));

    console.log(`✅ Pipeline Success! Successfully merged ${finalIndexedQuestions.length} targeted questions.`);

    res.json({
      success: true,
      subject: targetSubject,
      topic: topic || null,
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

    const startBrace = evaluationResultText.indexOf('{');
    const endBrace = evaluationResultText.lastIndexOf('}');
    if (startBrace === -1 || endBrace === -1) throw new Error("Evaluation Engine failed to output a reliable structured matrix response.");

    const finalEvaluatedPayload = JSON.parse(evaluationResultText.substring(startBrace, endBrace + 1));
    
    // 🚨 6. STRICT SCORE VALIDATION & BOUNDARY CLAMP SECURITY GATEWAY
    let scoreGiven = parseFloat(finalEvaluatedPayload.score_given);
    if (isNaN(scoreGiven)) scoreGiven = 0.0;
    scoreGiven = Math.min(parsedMaxMarks, Math.max(0.0, scoreGiven)); // Hard clamp validation fence

    console.log(`✅ Subjective Evaluation Complete! Core Score Compiled: ${scoreGiven}/${parsedMaxMarks}`);

    res.json({
      success: true,
      evaluation: {
        score_given: scoreGiven,
        ai_evaluation: {
          student_points: finalEvaluatedPayload.ai_evaluation?.student_points || ["Points evaluated contextually."],
          scope_of_improvement: finalEvaluatedPayload.ai_evaluation?.scope_of_improvement || "Refine formatting matrices layouts."
        }
      }
    });

  } catch (error) {
    console.error("❌ Subjective Evaluator Pipeline Error:", error);
    res.status(500).json({
      success: false,
      error: error.message || "An error locked up the subjective processing module matrix."
    });
  }
});

app.listen(PORT, () => console.log(`🔥 Production Secure Server running active on port: ${PORT}`));