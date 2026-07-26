import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';

dotenv.config();

// 🚨 1. FAIL-FAST STARTUP LAYER
if (!process.env.GEMINI_API_KEY) {
  console.error("❌ CRITICAL BOOT FAILURE: GEMINI_API_KEY environment variable is missing inside backend .env file!");
  process.exit(1);
}
if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
  console.error("❌ CRITICAL BOOT FAILURE: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing inside backend .env file!");
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

// 🔑 SUPABASE SERVICE ROLE CLIENT
// This client uses the service_role key, which BYPASSES all Row Level Security.
// This is intentional and safe here because:
//   1. This client only ever lives on the backend server, never sent to the browser.
//   2. question_pool and attempts_ledger have RLS enabled with ZERO policies —
//      meaning the frontend (using its anon/authenticated key) cannot touch them at all.
//      This backend is the ONLY door into those two tables.
//   3. Answer-hiding (correct_option_index, correct_answer, explanation) is enforced
//      here in code before sending any response to the frontend — not by RLS,
//      since RLS can only block/allow whole rows, not individual columns.
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

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
// (UNCHANGED — kept exactly as-is, still used wherever pool-based
//  serving isn't wired in yet, or for one-off generation needs)
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
          parsedData = JSON.parse(responseText.substring(startBrace, endBrace + 1));
          break;
        } catch (err) {
          retries--;
          console.warn(`⚠️ Retry triggered on generation chunk. Remaining retries: ${retries}`);
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
// (UNCHANGED)
// ======================================================================
app.post('/api/evaluate-subjective', async (req, res) => {
  try {
    const { question, userAnswer, uploadedFiles, testTitle, maxMarks, studentId, questionId } = req.body;

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

    // 🎯 POOL LEDGER UPDATE (optional — only when this question came from the
    // shared question pool). Subjective questions don't have a strict
    // right/wrong answer, so "correct" for resurfacing purposes is defined
    // as scoring at least 30% of the max marks. Below that, the question
    // stays eligible to resurface (mirrors wrong-answer resurfacing for MCQs).
    if (studentId && questionId) {
      try {
        const passThreshold = 0.3;
        const isCorrect = parsedMaxMarks > 0 ? (scoreGiven / parsedMaxMarks) >= passThreshold : false;

        await supabase
          .from('attempts_ledger')
          .upsert(
            {
              student_id: studentId,
              question_id: questionId,
              is_correct: isCorrect,
              attempted_at: new Date().toISOString()
            },
            { onConflict: 'student_id,question_id' }
          );
      } catch (ledgerErr) {
        // Never let ledger bookkeeping block the student from getting their score back.
        console.error("⚠️ Subjective ledger update failed (non-blocking):", ledgerErr);
      }
    }

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

// ======================================================================
// 🧠 HELPER: Simple word-overlap similarity check (Similarity Safety Net)
// Not semantic — a lightweight Jaccard-style overlap on normalized words.
// Used to catch a freshly generated question that's just a reworded
// duplicate of something already in the pool for the same tag combo.
// ======================================================================
// ======================================================================
// 🧠 HELPER: Normalize tag fields (exam, subject, topic) so the pool
// treats "SSC CGL", "ssc cgl", and "SSC   CGL" as the exact same tag.
// Postgres text equality is case-sensitive by default, so without this,
// casing drift alone would silently fragment the pool and quietly bring
// back the repetition/duplication problem this whole system exists to fix.
// Only affects internal matching tags — never touches question text itself.
// ======================================================================
function normalizeTag(value) {
  if (!value || typeof value !== 'string') return value;
  return value.trim().replace(/\s+/g, ' ').toUpperCase();
}

function normalizeToWords(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .split(/\s+/)
    .filter(Boolean);
}

function wordOverlapRatio(wordsA, wordsB) {
  const setA = new Set(wordsA);
  const setB = new Set(wordsB);
  let overlap = 0;
  for (const w of setA) if (setB.has(w)) overlap++;
  const union = new Set([...setA, ...setB]).size;
  return union === 0 ? 0 : overlap / union;
}

const SIMILARITY_THRESHOLD = 0.75; // tune this later if it's too strict/loose in practice

function isLikelyDuplicate(newQuestion, existingEntries) {
  const newCombinedText = `${newQuestion.question || ""} ${(newQuestion.options || []).join(" ")}`;
  const newWords = normalizeToWords(newCombinedText);

  for (const existing of existingEntries) {
    const existingCombinedText = `${existing.question_text || ""} ${Array.isArray(existing.options) ? existing.options.join(" ") : ""}`;
    const existingWords = normalizeToWords(existingCombinedText);
    if (wordOverlapRatio(newWords, existingWords) >= SIMILARITY_THRESHOLD) return true;
  }
  return false;
}

// ======================================================================
// 🧠 HELPER: Fresh AI generation WITH an exclusion list baked into the
// prompt, so Gemini is proactively told what already exists in the pool
// for this exact tag combination (Exclusion List — Layer 1 of Section 6).
// ======================================================================
async function generateFreshQuestionsForPool({ targetExam, targetSubject, targetTopic, diffLevel, qType, lang, count, exclusionCandidates }) {
  const model = genAI.getGenerativeModel({
    model: "gemini-3.1-flash-lite",
    generationConfig: { responseMimeType: "application/json", temperature: 1.3 }
  });

  const topicFocusPhrase = targetTopic
    ? `Specific Topic Focus: "${targetTopic}".`
    : `No specific narrow topic given — cover general questions broadly across this subject/section.`;

  let exclusionBlock = "";
  if (exclusionCandidates && exclusionCandidates.length > 0) {
    const exclusionLines = exclusionCandidates.slice(0, 100).map((q, i) => {
      const optsText = Array.isArray(q.options) ? ` Options: ${q.options.join(' | ')}` : '';
      return `${i + 1}. ${q.question_text}${optsText}`;
    }).join('\n');
    exclusionBlock = `\n\n[DO NOT REPEAT — EXISTING QUESTIONS IN POOL]\nThese questions already exist for this exact exam/subject/topic/difficulty combination. Do NOT generate anything testing the same underlying concept, even if reworded differently:\n${exclusionLines}\n`;
  }

  const MAX_CHUNK_SIZE = 15;
  let allCompiled = [];
  let remaining = count;

  while (remaining > 0) {
    const chunkSize = Math.min(MAX_CHUNK_SIZE, remaining);
    const sessionSeed = Math.random().toString(36).substring(7);

    let prompt = "";
    if (qType === 'Subjective') {
      prompt = `Generate EXACTLY ${chunkSize} distinct descriptive/subjective questions for ${targetExam}. Subject/Section: "${targetSubject}". ${topicFocusPhrase} Lang: ${lang}. Seed: ${sessionSeed}
${exclusionBlock}
[STRICT COUNT CONSTRAINT]: Your JSON response array MUST contain exactly ${chunkSize} question object(s) inside the "questions" array.

[DIFFICULTY CALIBRATION]: Strict Enforcement for "${diffLevel}" level, matching the actual standard core papers of ${targetExam}.

JSON schema: {"questions": [{"question":"","explanation":""}]}.
Explanation: Max 35 words core grading framework points (used only as an internal marking-guideline hint, never shown to the student before they answer).`;
    } else {
      prompt = `Generate EXACTLY ${chunkSize} unique Multiple Choice Questions (MCQs) for ${targetExam}. Subject/Section: "${targetSubject}". ${topicFocusPhrase} Lang: ${lang}. Seed: ${sessionSeed}
${exclusionBlock}
[STRICT COUNT CONSTRAINT]: Your JSON response array MUST contain exactly ${chunkSize} question objects inside the "questions" array.

[DIFFICULTY CALIBRATION]: Strict Enforcement for "${diffLevel}" level, matching the actual standard core papers of ${targetExam}. No basic/textbook-direct questions unless difficulty is explicitly Easy.

[OPTIONS]: Distribute 'correctOptionIndex' randomly across 0,1,2,3.
JSON schema: {"questions": [{"question":"","options":["","","",""],"correctOptionIndex":0,"explanation":""}]}.
Explanation: Max 20 words core fact.`;
    }

    let retries = 3;
    let parsedData = null;

    while (retries > 0) {
      try {
        const result = await model.generateContent(prompt);
        const response = await result.response;
        const responseText = response.text();
        const startBrace = responseText.indexOf('{');
        const endBrace = responseText.lastIndexOf('}');
        if (startBrace === -1 || endBrace === -1) throw new Error("Invalid AI response structure.");
        parsedData = JSON.parse(responseText.substring(startBrace, endBrace + 1));
        break;
      } catch (err) {
        retries--;
        if (retries === 0) throw err;
        await new Promise(r => setTimeout(r, 2000));
      }
    }

    if (parsedData?.questions?.length) {
      allCompiled = [...allCompiled, ...parsedData.questions];
    }
    remaining -= chunkSize;
  }

  return allCompiled.slice(0, count);
}

// ======================================================================
// 🎯 ROUTE 3 (NEW): SERVE QUESTIONS FROM POOL (with AI fallback)
// Pool-first, AI-fallback-with-exclusion-list design.
// Answers are ALWAYS stripped before the response leaves this route —
// this is where "Layer 2" of the answer-hiding security actually lives.
// ======================================================================
app.post('/api/pool/serve-questions', async (req, res) => {
  try {
    const { studentId, exam, subject, topic, difficulty, type, count, language, origin } = req.body;

    if (!studentId) return res.status(400).json({ success: false, error: "studentId missing bhai!" });
    if (!subject) return res.status(400).json({ success: false, error: "Subject/Section missing bhai!" });

    const targetExam = normalizeTag(exam) || "COMPETITIVE EXAM";
    const targetSubject = normalizeTag(subject);
    const targetTopic = normalizeTag(topic) || null;
    const diffLevel = difficulty || "Medium";
    const qType = type || "Objective";
    const lang = language || "English";
    const rawRequested = parseInt(count) || 5;
    const totalRequested = Math.min(50, Math.max(3, rawRequested));

    // STEP 1: Pull candidate pool rows matching this exact tag combination.
    let poolQuery = supabase
      .from('question_pool')
      .select('*')
      .eq('target_exam', targetExam)
      .eq('subject', targetSubject)
      .eq('difficulty', diffLevel)
      .eq('type', qType);

    poolQuery = targetTopic ? poolQuery.eq('topic', targetTopic) : poolQuery.is('topic', null);

    const { data: candidatePool, error: poolErr } = await poolQuery.limit(500);
    if (poolErr) throw poolErr;

    // STEP 2: Check this student's ledger for these specific candidate questions,
    // so we know which are "already correct" (exclude) vs "previously wrong" (prioritize) vs "unseen".
    const poolIds = (candidatePool || []).map(q => q.id);
    let ledgerRows = [];
    if (poolIds.length > 0) {
      const { data: ledgerData, error: ledgerErr } = await supabase
        .from('attempts_ledger')
        .select('question_id, is_correct')
        .eq('student_id', studentId)
        .in('question_id', poolIds);
      if (ledgerErr) throw ledgerErr;
      ledgerRows = ledgerData || [];
    }

    const correctIds = new Set(ledgerRows.filter(r => r.is_correct).map(r => r.question_id));
    const incorrectIds = new Set(ledgerRows.filter(r => !r.is_correct).map(r => r.question_id));

    const incorrectQuestions = candidatePool.filter(q => incorrectIds.has(q.id));   // resurface priority
    const unseenQuestions = candidatePool.filter(q => !correctIds.has(q.id) && !incorrectIds.has(q.id));

    let selected = [...incorrectQuestions, ...unseenQuestions].slice(0, totalRequested);
    const shortfall = totalRequested - selected.length;

    let freshlyGeneratedCount = 0;

    // STEP 3: AI fallback only for the shortfall, with exclusion list + similarity safety net.
    if (shortfall > 0) {
      const { data: exclusionCandidates, error: exclErr } = await supabase
        .from('question_pool')
        .select('question_text, options')
        .eq('target_exam', targetExam)
        .eq('subject', targetSubject)
        .eq('difficulty', diffLevel)
        .eq('type', qType)
        .order('created_at', { ascending: false })
        .limit(100);
      if (exclErr) throw exclErr;

      const freshQuestions = await generateFreshQuestionsForPool({
        targetExam, targetSubject, targetTopic, diffLevel, qType, lang,
        count: shortfall,
        exclusionCandidates: exclusionCandidates || []
      });

      const alreadyCheckedThisBatch = [...(exclusionCandidates || [])];

      for (const q of freshQuestions) {
        if (isLikelyDuplicate(q, alreadyCheckedThisBatch)) {
          console.warn("⚠️ Discarded a freshly generated question — flagged as likely duplicate by similarity safety net.");
          continue;
        }

        const { data: inserted, error: insertErr } = await supabase
          .from('question_pool')
          .insert({
            target_exam: targetExam,
            subject: targetSubject,
            topic: targetTopic,
            difficulty: diffLevel,
            type: qType,
            question_text: q.question,
            options: q.options || null,
            correct_option_index: typeof q.correctOptionIndex === 'number' ? q.correctOptionIndex : null,
            explanation: q.explanation || null,
            origin_note: origin || 'unspecified'
          })
          .select()
          .single();

        if (insertErr) {
          console.error("❌ Failed to insert freshly generated question into pool:", insertErr);
          continue;
        }

        alreadyCheckedThisBatch.push(inserted);
        selected.push(inserted);
        freshlyGeneratedCount++;
      }
    }

    // STEP 4: Strip answers before this ever leaves the backend.
    const safeQuestions = selected.slice(0, totalRequested).map(q => ({
      id: q.id,
      question: q.question_text,
      options: q.options,
      type: q.type
      // correct_option_index, correct_answer, explanation deliberately NOT included here
    }));

    res.json({
      success: true,
      questions: safeQuestions,
      meta: {
        servedFromPool: safeQuestions.length - freshlyGeneratedCount,
        freshlyGenerated: freshlyGeneratedCount
      }
    });

  } catch (error) {
    console.error("❌ Pool Serve Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to serve questions from pool." });
  }
});

// ======================================================================
// 🎯 ROUTE 4 (NEW): SUBMIT AN ATTEMPT
// Upserts into attempts_ledger keyed on (student_id, question_id) —
// same row gets updated on retry, never duplicated, thanks to the
// unique constraint on the table. Only place correct_option_index /
// explanation get sent back to the frontend is AFTER this call.
// ======================================================================
app.post('/api/pool/submit-attempt', async (req, res) => {
  try {
    const { studentId, questionId, selectedOptionIndex } = req.body;

    if (!studentId || !questionId) {
      return res.status(400).json({ success: false, error: "studentId or questionId missing." });
    }

    const { data: questionRow, error: qErr } = await supabase
      .from('question_pool')
      .select('*')
      .eq('id', questionId)
      .single();

    if (qErr || !questionRow) {
      return res.status(404).json({ success: false, error: "Question not found in pool." });
    }

    // Subjective questions are graded via /api/evaluate-subjective separately —
    // this route only handles Objective correctness right now.
    const isCorrect = questionRow.type === 'Objective'
      ? parseInt(selectedOptionIndex) === questionRow.correct_option_index
      : false;

    // Upsert: only touches student_id, question_id, is_correct, attempted_at.
    // The 'saved' column is deliberately NOT included here, so an existing
    // saved=true from a previous attempt is never overwritten back to false.
    const { data: upserted, error: upsertErr } = await supabase
      .from('attempts_ledger')
      .upsert(
        {
          student_id: studentId,
          question_id: questionId,
          is_correct: isCorrect,
          attempted_at: new Date().toISOString()
        },
        { onConflict: 'student_id,question_id' }
      )
      .select()
      .single();

    if (upsertErr) throw upsertErr;

    res.json({
      success: true,
      is_correct: isCorrect,
      correct_option_index: questionRow.correct_option_index,
      explanation: questionRow.explanation
    });

  } catch (error) {
    console.error("❌ Submit Attempt Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to submit attempt." });
  }
});

// ======================================================================
// 🎯 ROUTE 5 (NEW): TOGGLE SAVE
// Save is only allowed on a question that's already been attempted
// (product decision) — so this just updates the existing ledger row,
// it never creates one.
// ======================================================================
app.post('/api/pool/toggle-save', async (req, res) => {
  try {
    const { studentId, questionId, saved } = req.body;

    if (!studentId || !questionId || typeof saved !== 'boolean') {
      return res.status(400).json({ success: false, error: "studentId, questionId and saved (boolean) are all required." });
    }

    const { data, error } = await supabase
      .from('attempts_ledger')
      .update({ saved })
      .eq('student_id', studentId)
      .eq('question_id', questionId)
      .select()
      .single();

    if (error || !data) {
      return res.status(404).json({ success: false, error: "No attempt found for this question — it must be attempted before it can be saved." });
    }

    res.json({ success: true, saved: data.saved });

  } catch (error) {
    console.error("❌ Toggle Save Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to toggle save state." });
  }
});

// ======================================================================
// 🎯 ROUTE 6: BUILD A FULL TEST PAPER FROM THE POOL (with AI fallback)
// Used by AI Labs. As of "Secure Test Delivery", this route strips
// answers from its response just like /api/pool/serve-questions —
// correctness is only ever revealed after an attempt, via
// /api/pool/grade-test. Full data is still written to question_pool
// (server-side only) so grading has something to check against.
// Supports both Objective and Subjective question types.
// ======================================================================
app.post('/api/pool/build-test', async (req, res) => {
  try {
    const { studentId, exam, subject, topic, difficulty, type, count, language, origin, revealAnswers } = req.body;

    if (!studentId) return res.status(400).json({ success: false, error: "studentId missing bhai!" });
    if (!subject) return res.status(400).json({ success: false, error: "Subject/Section missing bhai!" });

    const targetExam = normalizeTag(exam) || "COMPETITIVE EXAM";
    const targetSubject = normalizeTag(subject);
    const targetTopic = normalizeTag(topic) || null;
    const diffLevel = difficulty || "Medium";
    const qType = type || "Objective";
    const lang = language || "English";
    const rawRequested = parseInt(count) || 5;
    const totalRequested = Math.min(50, Math.max(1, rawRequested));

    // STEP 1: Pull candidate pool rows matching this exact tag combination.
    let poolQuery = supabase
      .from('question_pool')
      .select('*')
      .eq('target_exam', targetExam)
      .eq('subject', targetSubject)
      .eq('difficulty', diffLevel)
      .eq('type', qType);

    poolQuery = targetTopic ? poolQuery.eq('topic', targetTopic) : poolQuery.is('topic', null);

    const { data: candidatePool, error: poolErr } = await poolQuery.limit(500);
    if (poolErr) throw poolErr;

    // STEP 2: Ledger check — same resurfacing rule as serve-questions.
    const poolIds = (candidatePool || []).map(q => q.id);
    let ledgerRows = [];
    if (poolIds.length > 0) {
      const { data: ledgerData, error: ledgerErr } = await supabase
        .from('attempts_ledger')
        .select('question_id, is_correct')
        .eq('student_id', studentId)
        .in('question_id', poolIds);
      if (ledgerErr) throw ledgerErr;
      ledgerRows = ledgerData || [];
    }

    const correctIds = new Set(ledgerRows.filter(r => r.is_correct).map(r => r.question_id));
    const incorrectIds = new Set(ledgerRows.filter(r => !r.is_correct).map(r => r.question_id));

    const incorrectQuestions = candidatePool.filter(q => incorrectIds.has(q.id));
    const unseenQuestions = candidatePool.filter(q => !correctIds.has(q.id) && !incorrectIds.has(q.id));

    let selected = [...incorrectQuestions, ...unseenQuestions].slice(0, totalRequested);
    const shortfall = totalRequested - selected.length;

    let freshlyGeneratedCount = 0;

    if (shortfall > 0) {
      const { data: exclusionCandidates, error: exclErr } = await supabase
        .from('question_pool')
        .select('question_text, options')
        .eq('target_exam', targetExam)
        .eq('subject', targetSubject)
        .eq('difficulty', diffLevel)
        .eq('type', qType)
        .order('created_at', { ascending: false })
        .limit(100);
      if (exclErr) throw exclErr;

      const freshQuestions = await generateFreshQuestionsForPool({
        targetExam, targetSubject, targetTopic, diffLevel, qType, lang,
        count: shortfall,
        exclusionCandidates: exclusionCandidates || []
      });

      const alreadyCheckedThisBatch = [...(exclusionCandidates || [])];

      for (const q of freshQuestions) {
        if (isLikelyDuplicate(q, alreadyCheckedThisBatch)) {
          console.warn("⚠️ Discarded a freshly generated question — flagged as likely duplicate by similarity safety net.");
          continue;
        }

        const insertPayload = {
          target_exam: targetExam,
          subject: targetSubject,
          topic: targetTopic,
          difficulty: diffLevel,
          type: qType,
          question_text: q.question,
          origin_note: origin || 'unspecified'
        };

        if (qType === 'Objective') {
          insertPayload.options = q.options || null;
          insertPayload.correct_option_index = typeof q.correctOptionIndex === 'number' ? q.correctOptionIndex : null;
          insertPayload.explanation = q.explanation || null;
        } else {
          // Subjective: no options/correct_option_index — graded later via AI evaluation.
          insertPayload.explanation = q.explanation || null; // internal marking-guideline hint only
        }

        const { data: inserted, error: insertErr } = await supabase
          .from('question_pool')
          .insert(insertPayload)
          .select()
          .single();

        if (insertErr) {
          console.error("❌ Failed to insert freshly generated question into pool:", insertErr);
          continue;
        }

        alreadyCheckedThisBatch.push(inserted);
        selected.push(inserted);
        freshlyGeneratedCount++;
      }
    }

    // STEP 3: Strip answers by default ("Secure Test Delivery") — correctness
    // is only revealed post-attempt via /api/pool/grade-test. EXCEPTION:
    // BrainFeed explicitly passes revealAnswers=true, since it's a casual
    // practice quiz (not a timed exam) and wants instant feedback — that's
    // a deliberate, informed trade-off, not an oversight.
    const fullQuestions = selected.slice(0, totalRequested).map(q => {
      const base = { id: q.id, question: q.question_text, options: q.options, type: q.type };
      if (revealAnswers === true) {
        base.correctOptionIndex = q.correct_option_index;
        base.explanation = q.explanation;
      }
      return base;
    });

    res.json({
      success: true,
      questions: fullQuestions,
      meta: {
        servedFromPool: fullQuestions.length - freshlyGeneratedCount,
        freshlyGenerated: freshlyGeneratedCount
      }
    });

  } catch (error) {
    console.error("❌ Build Test Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to build test from pool." });
  }
});

// ======================================================================
// 🎯 ROUTE 7 (NEW): GET SAVED QUESTIONS
// Powers Library.jsx (full saved-question details) and Statistics.jsx
// (saved-question count). Joins attempts_ledger -> question_pool so the
// full question text/options/explanation come back in one call — safe to
// show full answer data here since these are ALWAYS post-attempt saves.
// ======================================================================
app.get('/api/pool/saved-questions', async (req, res) => {
  try {
    const { studentId } = req.query;

    if (!studentId) return res.status(400).json({ success: false, error: "studentId missing bhai!" });

    const { data, error } = await supabase
      .from('attempts_ledger')
      .select('question_id, is_correct, attempted_at, question_pool(question_text, options, correct_option_index, explanation, target_exam, subject, topic, type)')
      .eq('student_id', studentId)
      .eq('saved', true)
      .order('attempted_at', { ascending: false });

    if (error) throw error;

    const savedQuestions = (data || [])
      .filter(row => row.question_pool) // guard against a question that was deleted from the pool after being saved
      .map(row => ({
        id: row.question_id,
        question: row.question_pool.question_text,
        options: row.question_pool.options,
        correctOptionIndex: row.question_pool.correct_option_index,
        explanation: row.question_pool.explanation,
        exam: row.question_pool.target_exam,
        subject: row.question_pool.subject,
        topic: row.question_pool.topic,
        type: row.question_pool.type,
        wasCorrect: row.is_correct,
        savedAt: row.attempted_at
      }));

    res.json({ success: true, savedQuestions, count: savedQuestions.length });

  } catch (error) {
    console.error("❌ Saved Questions Fetch Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to fetch saved questions." });
  }
});

// ======================================================================
// 🎯 ROUTE 8 (NEW): GRADE A TEST — "Secure Test Delivery"
// TestPortal sends back {questionId, selectedOptionIndex, marks, neg} for
// every attempted Objective question — NEVER the answer itself, since it
// was never sent to the browser to begin with. This route resolves the
// real answer server-side:
//   1. Try question_pool first (covers ALL AI Labs questions — fresh or
//      published to mock_tests, since their id is always a real pool UUID).
//   2. Fall back to the mock_tests row's embedded questions_list (covers
//      Custom Builder-authored questions, which never touch the pool by
//      design — admin owns those answers directly).
// Also logs pool ledger entries for anything resolved via the pool.
// ======================================================================
app.post('/api/pool/grade-test', async (req, res) => {
  try {
    const { studentId, testId, answers } = req.body;

    if (!Array.isArray(answers) || answers.length === 0) {
      return res.json({ success: true, results: [], totalScore: 0, correctCount: 0, incorrectCount: 0 });
    }

    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    const candidatePoolIds = answers.map(a => a.questionId).filter(id => uuidRegex.test(String(id)));

    // STEP 1: Resolve as many as possible from question_pool.
    let poolAnswerMap = {};
    if (candidatePoolIds.length > 0) {
      const { data: poolRows, error: poolErr } = await supabase
        .from('question_pool')
        .select('id, correct_option_index, explanation')
        .in('id', candidatePoolIds);
      if (poolErr) throw poolErr;
      (poolRows || []).forEach(row => {
        poolAnswerMap[row.id] = { correctOptionIndex: row.correct_option_index, explanation: row.explanation };
      });
    }

    // STEP 2: Anything not resolved via the pool falls back to the
    // mock_tests row's own embedded questions_list (Custom Builder path).
    const unresolvedIds = answers.map(a => String(a.questionId)).filter(id => !poolAnswerMap[id]);
    let mockTestsAnswerMap = {};
    if (unresolvedIds.length > 0 && testId) {
      const { data: mockRow, error: mockErr } = await supabase
        .from('mock_tests')
        .select('questions_list')
        .eq('id', testId)
        .maybeSingle();
      if (!mockErr && mockRow && Array.isArray(mockRow.questions_list)) {
        mockRow.questions_list.forEach(q => {
          if (unresolvedIds.includes(String(q.id))) {
            const correctVal = q.correct !== undefined ? q.correct : q.correctOptionIndex;
            mockTestsAnswerMap[String(q.id)] = { correctOptionIndex: correctVal, explanation: q.explanation };
          }
        });
      }
    }

    // STEP 3: Score every answer + build the ledger update list.
    let totalScore = 0;
    let correctCount = 0;
    let incorrectCount = 0;
    const results = [];
    const ledgerUpserts = [];

    for (const ans of answers) {
      const qid = String(ans.questionId);
      const wasAttempted = ans.selectedOptionIndex !== null && ans.selectedOptionIndex !== undefined && ans.selectedOptionIndex !== "";
      const source = poolAnswerMap[qid] || mockTestsAnswerMap[qid];
      const correctOptionIndex = source ? source.correctOptionIndex : null;
      const explanation = source ? source.explanation : null;
      const selectedOptionIndex = wasAttempted ? parseInt(ans.selectedOptionIndex) : null;
      const hasResolvedAnswer = correctOptionIndex !== null && correctOptionIndex !== undefined;
      const isCorrect = wasAttempted && hasResolvedAnswer && selectedOptionIndex === parseInt(correctOptionIndex);

      const marksVal = parseFloat(String(ans.marks || '2.0').replace('+', '')) || 0;
      const negVal = parseFloat(String(ans.neg || '0.66').replace('-', '')) || 0;
      // Skipped questions never affect score or correct/incorrect counts —
      // they're only included here so the analysis screen can still show
      // the right answer for a question the student didn't attempt.
      const marksAwarded = (!wasAttempted || !hasResolvedAnswer) ? 0 : (isCorrect ? marksVal : -negVal);

      totalScore += marksAwarded;
      if (wasAttempted && hasResolvedAnswer) {
        if (isCorrect) correctCount++; else incorrectCount++;
      }

      results.push({ questionId: qid, isCorrect, correctOptionIndex, explanation, marksAwarded, wasAttempted });

      // Only pool-resolved, ACTUALLY ATTEMPTED questions get logged to the
      // shared ledger — Custom Builder questions were never in the pool,
      // and skipped questions shouldn't count as "seen and mastered/missed".
      if (studentId && wasAttempted && poolAnswerMap[qid]) {
        ledgerUpserts.push({
          student_id: studentId,
          question_id: qid,
          is_correct: isCorrect,
          attempted_at: new Date().toISOString()
        });
      }
    }

    if (ledgerUpserts.length > 0) {
      const { error: ledgerErr } = await supabase
        .from('attempts_ledger')
        .upsert(ledgerUpserts, { onConflict: 'student_id,question_id' });
      if (ledgerErr) console.error("⚠️ Batch ledger update failed (non-blocking):", ledgerErr);
    }

    res.json({
      success: true,
      results,
      totalScore: parseFloat(totalScore.toFixed(2)),
      correctCount,
      incorrectCount
    });

  } catch (error) {
    console.error("❌ Grade Test Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to grade test." });
  }
});

app.listen(PORT, () => console.log(`🔥 Production Secure Server running active on port: ${PORT}`));