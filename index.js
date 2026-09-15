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

// 🛡️ AUTH VERIFICATION MIDDLEWARE
// This replaces trusting a client-sent `studentId` in the request body.
// Frontend must send: Authorization: Bearer <supabase_session_access_token>
// This middleware verifies that token with Supabase itself, and only then
// sets req.verifiedUserId to the REAL user id — which the rest of the route
// handler must use instead of req.body.studentId.
async function requireAuth(req, res, next) {
  try {
    const authHeader = req.headers.authorization || '';
    const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;

    if (!token) {
      return res.status(401).json({ success: false, error: "Login required (missing auth token)." });
    }

    // Ask Supabase: "who does this token actually belong to?"
    const { data, error } = await supabase.auth.getUser(token);

    if (error || !data?.user) {
      return res.status(401).json({ success: false, error: "Invalid or expired session. Please log in again." });
    }

    // This is now the ONLY source of truth for user identity in this request.
    req.verifiedUserId = data.user.id;
    next();
  } catch (err) {
    console.error("Auth middleware error:", err);
    return res.status(401).json({ success: false, error: "Authentication check failed." });
  }
}

// 🚦 PER-MINUTE RATE LIMITING (in-memory, per verified user)
// This is a burst-friendly cap — NOT a per-call cooldown. A single user
// action (e.g. generating a 100-question paper, or submitting a test with
// many Subjective questions) can legitimately fire several sequential
// batch calls in a few seconds. A flat "N seconds between any two calls"
// cooldown would make that impossible. Instead we cap total calls to a
// route within a rolling 60-second window per user — bursts are fine,
// sustained abuse is not. Pre-paid model: the student has already paid
// for their credit balance before any of these routes run, so this exists
// to protect infra/Gemini-side rate limits, not to prevent revenue loss.
// In-memory is sufficient here — if the server restarts, counters simply
// reset, which is an acceptable (and rare) edge case for a soft cap.
const rateLimitBuckets = new Map(); // key: `${routeName}:${userId}` -> { count, windowStart }

function createPerMinuteRateLimit(routeName, maxCallsPerMinute) {
  return function rateLimit(req, res, next) {
    const userId = req.verifiedUserId;
    const key = `${routeName}:${userId}`;
    const now = Date.now();
    const windowMs = 60 * 1000;

    const bucket = rateLimitBuckets.get(key);

    if (!bucket || now - bucket.windowStart >= windowMs) {
      // Fresh window.
      rateLimitBuckets.set(key, { count: 1, windowStart: now });
      return next();
    }

    if (bucket.count >= maxCallsPerMinute) {
      const secondsLeft = Math.ceil((windowMs - (now - bucket.windowStart)) / 1000);
      return res.status(429).json({
        success: false,
        error: `Too many requests — please wait ${secondsLeft}s before trying again.`
      });
    }

    bucket.count += 1;
    next();
  };
}

const rateLimitBuildTest = createPerMinuteRateLimit('build-test', 20);
const rateLimitEvaluateSubjective = createPerMinuteRateLimit('evaluate-subjective', 10);

// 🔒 IDEMPOTENCY LOCK for evaluate-subjective (prevents duplicate-submit
// spam — e.g. a double-tapped Submit button, or a network retry — from
// triggering a second Gemini evaluation for the SAME attempt while the
// first one is still being processed). Keyed on attemptId, since a whole
// batch of subjective questions for one submit shares one attemptId.
const processingAttempts = new Set();

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
app.post('/api/generate-test', requireAuth, async (req, res) => {
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
        
        [LATEX FORMATTING — MANDATORY, NOT OPTIONAL, WHENEVER MATH NOTATION APPEARS]: Any time your question or options text contains a mathematical symbol, you MUST wrap that entire expression in single dollar signs ($...$) using proper LaTeX commands — never output raw ASCII math shorthand. Concretely:
        - Powers/exponents: write $x^2$, $\\tan^2\\theta$ — NEVER "x^2" or "tan^2θ" as bare text outside $...$.
        - Roots: write $\\sqrt{x}$, $\\sqrt{y^2+2xy}$ — NEVER "√(y^2+2xy)" as bare text.
        - Fractions: write $\\frac{a}{b}$ — NEVER "a/b" when it represents a real fraction/ratio.
        - Greek letters/trig: write $\\theta$, $\\lambda$, $\\sin\\theta$, $\\cos\\theta$, $\\cot\\theta$ — never bare Unicode glyphs.
        - Subscripts: write $x_1$, $a_n$ — never "x_1" as plain text.
        This applies to BOTH the "question" field AND every string inside "options". Do NOT force LaTeX into purely conceptual, non-mathematical humanities/geography text — but the instant ANY exponent, root, fraction-as-ratio, or Greek/trig symbol appears anywhere, wrapping it is a hard requirement with zero exceptions.
        
        [JSON ESCAPE SAFETY — CRITICAL, APPLIES TO YOUR RAW OUTPUT]: Your entire response is parsed as JSON. Any backslash you write inside a JSON string value MUST be a DOUBLE backslash (\\\\), never a single one — this applies to every LaTeX command (write \\\\frac not \\frac, \\\\sqrt not \\sqrt, \\\\theta not \\theta) and to any other backslash-like character. A single backslash before certain letters (like f, n, t, b, r) breaks JSON parsing or silently corrupts your own output. Also, for non-LaTeX content (e.g. geography coordinates, degree-minute-second notation like 23°26'22", or apostrophes in words like "Earth's"), never insert a backslash before the apostrophe or quote character — write it as a plain straight character; do NOT write \\' or \\".
        
        [DIFFICULTY CALIBRATION]: Strict Enforcement for "${diffLevel}" level. If difficulty is "Medium", it must strictly match the actual standard core papers of ${targetExam}—make it highly conceptual, analytical, and tricky (ABSOLUTELY NO basic or direct textbook questions). If difficulty is "Tough", make it brutally advanced, elite level, requiring complex structural logic.
        
        [STRUCTURE]: For multi-statement, matching, or list-based questions, do NOT lump statements into one paragraph. You MUST format statements as a clean numbered vertical list (e.g., "Consider the following statements:\\n\\n1. [Statement 1]\\n\\n2. [Statement 2]") with explicit double escaped newlines (\\n\\n) after each item so the frontend renders them beautifully.
        
        [OPTIONS]: Distribute 'correctOptionIndex' randomly across 0,1,2,3.

        [SELF-VERIFICATION — MANDATORY BEFORE FINALIZING EACH QUESTION]: Before finalizing each question, work through this internally: (1) SOLVE: Actually derive the correct answer yourself from first principles for this specific question — compute it, don't assume it (for patterns/ciphers/sequences: apply the rule to EVERY single element/letter and confirm it holds for ALL of them, not just some; for math: do the actual calculation; for facts: confirm accuracy). (2) MATCH: Confirm your derived answer exactly equals the correctOptionIndex option's text/value — not approximately, exactly. (3) EXPLANATION CHECK: Confirm the explanation you are about to write, if followed literally step by step, actually produces YOUR derived answer and no other option. (4) DISTRACTORS: Confirm the 3 wrong options are plausible (reflect common mistakes) but are clearly wrong once your verified derivation is applied. (5) LATEX CHECK: Scan your own question and options text for any bare ^, √, Greek letter, or a/b-style fraction that is NOT wrapped in $...$ — if found, fix it before output. (6) If any check in (1)-(5) fails, discard this draft internally and construct a cleaner question from scratch — do not output anything that fails this verification. Only the final, verified question objects should appear in your output.

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
          const rawJsonSlice = responseText.substring(startBrace, endBrace + 1);
          parsedData = JSON.parse(sanitizeJsonEscapes(rawJsonSlice));
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
      ...applyLatexSafetyNet(q),
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
// 📝 ROUTE 2: MULTIMODAL SUBJECTIVE EVALUATION GATEWAY — BATCH MODE
// Accepts an ARRAY of subjective questions for one attempt and evaluates
// ALL of them in a SINGLE Gemini call, instead of one call per question.
// This matters because a submit with N Subjective questions used to fire
// N sequential Gemini calls — expensive, slow, and hard to rate-limit
// sanely (a burst of legitimate calls looks identical to abuse). Batching
// collapses that to 1 call per chunk (max 10 questions per chunk — see
// BATCH_SIZE_LIMIT below; the frontend chunks larger sets the same way
// AiTests.jsx already chunks build-test into groups of 15).
//
// Images: rather than sending base64 inline (which bloats the request and
// gains nothing extra from Gemini, since cost is token-based either way),
// each image is referenced via a short-lived signed URL from the private
// 'subjective-uploads' Storage bucket, passed to Gemini as a fileData
// part with a fileUri. Gemini fetches it directly — no base64 needed.
//
// Idempotency: the whole attempt is locked by attemptId while processing,
// so a double-submit or network retry can't trigger a second (paid-for)
// Gemini evaluation for work that's already in flight.
// ======================================================================
const BATCH_SIZE_LIMIT = 10;

app.post('/api/evaluate-subjective', requireAuth, rateLimitEvaluateSubjective, async (req, res) => {
  const studentId = req.verifiedUserId; // ✅ server-verified
  const { attemptId, testTitle, questions } = req.body;

  if (!attemptId) {
    return res.status(400).json({ success: false, error: "attemptId is required." });
  }
  if (!Array.isArray(questions) || questions.length === 0) {
    return res.status(400).json({ success: false, error: "questions must be a non-empty array." });
  }
  if (questions.length > BATCH_SIZE_LIMIT) {
    return res.status(400).json({ success: false, error: `Max ${BATCH_SIZE_LIMIT} questions per batch — please chunk larger sets.` });
  }

  // 🔒 Idempotency lock — reject if this exact attempt is already being evaluated.
  if (processingAttempts.has(attemptId)) {
    return res.status(409).json({ success: false, error: "This attempt is already being evaluated — please wait." });
  }
  processingAttempts.add(attemptId);

  try {
    const evaluationModel = genAI.getGenerativeModel({
      model: "gemini-3.1-flash-lite",
      generationConfig: { responseMimeType: "application/json" }
    });

    const computedTestTitle = testTitle || "Descriptive Assessment Challenge";

    // For each question, resolve any uploaded image's storage path into a
    // short-lived signed URL (bucket is private — Gemini needs a URL it
    // can actually fetch, not the bare path). Ownership isn't re-checked
    // here since these paths were written under this same student's own
    // {user_id}/... prefix at upload time moments earlier in this same
    // submit flow.
    const questionsWithSignedUrls = await Promise.all(questions.map(async (q) => {
      let signedImageUrls = [];
      if (Array.isArray(q.uploadedFiles)) {
        const results = await Promise.all(q.uploadedFiles.map(async (file) => {
          if (!file.path) return null;
          try {
            const { data, error } = await supabase.storage
              .from('subjective-uploads')
              .createSignedUrl(file.path, 60 * 5); // 5 minutes — plenty for immediate evaluation
            if (error || !data?.signedUrl) return null;
            return { url: data.signedUrl, mimeType: file.type || 'image/jpeg' };
          } catch {
            return null;
          }
        }));
        signedImageUrls = results.filter(Boolean);
      }
      return { ...q, signedImageUrls };
    }));

    const parsedQuestionsBlock = questionsWithSignedUrls.map((q, idx) => `
      --- Question ${idx + 1} (questionId: "${q.questionId}") ---
      Question Statement: "${q.question}"
      Maximum Marks: ${parseFloat(q.maxMarks) || 10.0}
      Student Text Answer: "${q.userAnswer || "None provided"}"
      ${q.signedImageUrls.length > 0 ? `(A handwritten answer image for this question follows below, in the same order.)` : `(No image provided for this question — evaluate from text answer alone.)`}
    `).join('\n');

    const batchEvaluationPrompt = `
      You are an elite, highly critical senior examiner executing rigorous assessments for Project Infinity.
      Test Context: "${computedTestTitle}"

      Below are ${questionsWithSignedUrls.length} separate subjective questions from the SAME test attempt.
      Evaluate EACH ONE independently — do not let one question's content influence another's score.

      [EVALUATION RULES]
      1. DYNAMIC CONTEXT ADAPTATION: adapt grading severity to the exam level implied by the test title:
         - Civil Services (UPSC, State PSC): multi-dimensional analysis, administrative alignment, logical flow.
         - Secondary School Boards (CBSE/ICSE): exact textual definitions, key terms, textbook compliance.
         - Other exams (SSC, Banking, Technical): factual accuracy, structure, to-the-point answers.
      2. EXTREME STRICTNESS: do not award marks casually. Deduct for poor structuring, vague concepts, missing references.
      3. SCALE-PROPORTIONAL SCORING: score_given must scale precisely between 0 and each question's own max marks.
      4. STRICT WORD LIMITS (critical — batch responses must stay compact):
         - "student_points": MAXIMUM 2 bullet points, each 15-20 words max.
         - "scope_of_improvement": ONE sentence, 40-50 words max.
         - Do not exceed these limits under any circumstance, even for a poor or blank answer.

      [QUESTIONS]
      ${parsedQuestionsBlock}

      [OUTPUT SCHEMA — RETURN EXACTLY THIS STRUCTURE, ONE ENTRY PER QUESTION, SAME ORDER]
      {
        "evaluations": [
          {
            "questionId": "<the exact questionId string given above>",
            "score_given": 0.0,
            "ai_evaluation": {
              "student_points": ["point 1 (max 20 words)", "point 2 (max 20 words)"],
              "scope_of_improvement": "max 50 words"
            }
          }
        ]
      }
    `;

    const generativePayloadParts = [batchEvaluationPrompt];
    for (const q of questionsWithSignedUrls) {
      for (const img of q.signedImageUrls) {
        generativePayloadParts.push({
          fileData: { fileUri: img.url, mimeType: img.mimeType }
        });
      }
    }

    let retries = 3;
    let evaluationResultText = "";

    while (retries > 0) {
      try {
        console.log(`🔍 Batch-evaluating ${questionsWithSignedUrls.length} subjective questions for attempt ${attemptId}...`);
        const result = await evaluationModel.generateContent(generativePayloadParts);
        const finalResponse = await result.response;
        evaluationResultText = finalResponse.text();
        break;
      } catch (err) {
        retries--;
        console.warn(`⚠️ Batch subjective evaluator hit an error. Retrying...`);
        if (retries === 0) throw err;
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }

    const startBrace = evaluationResultText.indexOf('{');
    const endBrace = evaluationResultText.lastIndexOf('}');
    if (startBrace === -1 || endBrace === -1) throw new Error("Evaluation engine failed to output a reliable structured response.");

    const finalPayload = JSON.parse(evaluationResultText.substring(startBrace, endBrace + 1));
    const rawEvaluations = Array.isArray(finalPayload.evaluations) ? finalPayload.evaluations : [];

    // Map back onto the ORIGINAL question list by questionId, so a missing
    // or malformed entry for one question doesn't break the whole batch —
    // it just falls back to a 0-score placeholder for that one question.
    const evaluationsById = new Map(rawEvaluations.map(e => [String(e.questionId), e]));

    const finalResults = questionsWithSignedUrls.map((q) => {
      const evalEntry = evaluationsById.get(String(q.questionId));
      const maxMarks = parseFloat(q.maxMarks) || 10.0;
      let scoreGiven = parseFloat(evalEntry?.score_given);
      if (isNaN(scoreGiven)) scoreGiven = 0.0;
      scoreGiven = Math.min(maxMarks, Math.max(0.0, scoreGiven));

      return {
        questionId: q.questionId,
        score_given: scoreGiven,
        ai_evaluation: {
          student_points: evalEntry?.ai_evaluation?.student_points || ["Points evaluated contextually."],
          scope_of_improvement: evalEntry?.ai_evaluation?.scope_of_improvement || "Refine structure and completeness."
        }
      };
    });

    // 🎯 POOL LEDGER UPDATE — best-effort, per question, non-blocking.
    for (const r of finalResults) {
      const originalQ = questionsWithSignedUrls.find(q => q.questionId === r.questionId);
      const maxMarks = parseFloat(originalQ?.maxMarks) || 10.0;
      try {
        const passThreshold = 0.3;
        const isCorrect = maxMarks > 0 ? (r.score_given / maxMarks) >= passThreshold : false;
        await supabase
          .from('attempts_ledger')
          .upsert(
            { student_id: studentId, question_id: r.questionId, is_correct: isCorrect, attempted_at: new Date().toISOString() },
            { onConflict: 'student_id,question_id' }
          );
      } catch (ledgerErr) {
        console.error("⚠️ Subjective ledger update failed (non-blocking):", ledgerErr);
      }
    }

    console.log(`✅ Batch evaluation complete for attempt ${attemptId}: ${finalResults.length} questions scored.`);
    res.json({ success: true, evaluations: finalResults });

  } catch (error) {
    console.error("❌ Batch Subjective Evaluator Error:", error);
    res.status(500).json({
      success: false,
      error: error.message || "An error occurred during batch subjective evaluation."
    });
  } finally {
    processingAttempts.delete(attemptId);
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

// ======================================================================
// 🧮 SAFETY-NET LATEX NORMALIZER
// The prompt instructs Gemini to wrap all math in $...$, but LLM prompt
// compliance is never 100% — this is a code-side backstop that catches the
// most common raw-math patterns it still occasionally leaks (bare "x^2",
// "√(...)", lone Greek letters) and auto-wraps them, so a missed instruction
// doesn't reach the student as unrendered "tan^2θ" plain text. This does NOT
// replace the prompt fix — it's a second line of defense for whatever slips
// through, since we can never fully guarantee generation-time compliance.
// ======================================================================
function autoWrapStrayLatex(text) {
  if (!text || typeof text !== 'string') return text;

  // Skip segments already inside $...$ so we never double-wrap.
  const segments = text.split(/(\$[^$]+\$)/g);

  return segments.map(seg => {
    if (seg.startsWith('$') && seg.endsWith('$')) return seg; // already LaTeX

    let fixed = seg;

    // Build a raw (un-dollar-wrapped) LaTeX form of a chunk of math text —
    // converts exponents and Greek letters to LaTeX commands WITHOUT adding
    // $ signs, so it's safe to nest inside \sqrt{...} or other wrappers
    // without producing invalid nested-$ LaTeX.
    const toRawLatex = (s) => {
      let r = s;
      r = r.replace(/([a-zA-Z0-9αβγθλμπφω]+)\^(\{[^}]+\}|[a-zA-Z0-9]+)/g, (_, base, exp) => {
        const cleanExp = exp.startsWith('{') ? exp : `{${exp}}`;
        return `${base}^${cleanExp}`;
      });
      r = r.replace(/([θλμπφωΔαβγ])/g, (match) => `\\${greekName(match)}`);
      return r;
    };

    // √(...) or √x  ->  $\sqrt{...}$ — process inner content for nested
    // exponents/Greek letters first, then wrap the WHOLE sqrt expression
    // in a single pair of $ (never nest $ inside $).
    fixed = fixed.replace(/√\(([^)]+)\)/g, (_, inner) => `$\\sqrt{${toRawLatex(inner)}}$`);
    fixed = fixed.replace(/√([a-zA-Z0-9]+)/g, (_, inner) => `$\\sqrt{${toRawLatex(inner)}}$`);

    // Re-split on the sqrt replacements we just made so we don't touch
    // their contents again in the passes below.
    const subSegments = fixed.split(/(\$[^$]+\$)/g);
    fixed = subSegments.map(sub => {
      if (sub.startsWith('$') && sub.endsWith('$')) return sub; // just-wrapped sqrt, leave alone

      // base^exponent (letters/digits, optional grouped exponent) -> $base^{exponent}$
      let s = sub.replace(/([a-zA-Z0-9αβγθλμπφω]+)\^(\{[^}]+\}|[a-zA-Z0-9]+)/g, (_, base, exp) => {
        const cleanExp = exp.startsWith('{') ? exp : `{${exp}}`;
        return `$${base}^${cleanExp}$`;
      });

      // Bare Greek letters used as math variables (θ, λ, π, etc.) standing alone
      s = s.replace(/([θλμπφωΔαβγ])/g, (match) => `$\\${greekName(match)}$`);

      return s;
    }).join('');

    return fixed;
  }).join('');
}

function greekName(ch) {
  const map = { 'θ': 'theta', 'λ': 'lambda', 'μ': 'mu', 'π': 'pi', 'φ': 'phi', 'ω': 'omega', 'Δ': 'Delta', 'α': 'alpha', 'β': 'beta', 'γ': 'gamma' };
  return map[ch] || ch;
}

// ======================================================================
// 🛡️ SAFETY NET: Sanitize broken backslash escapes in raw Gemini output
// BEFORE JSON.parse() runs.
//
// Why this exists: Gemini is instructed to output LaTeX (e.g. \frac{1}{2},
// \sqrt{x}, \theta) inside JSON string values. Valid JSON requires those
// backslashes to be DOUBLED (\\frac) so that after parsing, a single
// backslash survives. When Gemini forgets to double them, two failure
// modes happen:
//   1) The single backslash + next letter accidentally forms a JSON-
//      RESERVED escape (\f = form-feed, \t = tab, \n = newline, etc).
//      JSON.parse() silently "succeeds" but SWALLOWS the backslash and
//      that one letter, corrupting output — e.g. "\frac{1}{2}" parses
//      into "rac{1}{2}" because \f is consumed as a lone form-feed char.
//   2) The single backslash + next char is NOT a valid JSON escape at
//      all (e.g. a stray \' from an apostrophe). JSON.parse() throws
//      "Bad escaped character in JSON at position X" and the whole
//      generation call crashes (this is the AI Labs error case).
//
// NOTE: We deliberately do NOT try to blanket-detect "any backslash not
// already a valid JSON escape" — that's ambiguous, because \f, \t, \n etc.
// are simultaneously valid standalone JSON escapes AND the first letter of
// legitimate LaTeX commands (\frac, \tan, \newcommand...). There's no way
// to tell those apart generically. Instead we target the two KNOWN, real
// failure sources directly:
//   (a) A known finite list of LaTeX command names Gemini is instructed
//       to use — double the backslash only when one of these follows it.
//   (b) A stray \' (backslash-apostrophe), which is never valid JSON and
//       always represents an over-escaped apostrophe — the backslash is
//       simply dropped, restoring the plain apostrophe.
// ======================================================================
function sanitizeJsonEscapes(rawText) {
  if (typeof rawText !== 'string') return rawText;

  const knownLatexCommands = [
    'dfrac', 'frac', 'sqrt', 'theta', 'lambda', 'omega', 'Omega', 'Delta',
    'delta', 'alpha', 'beta', 'gamma', 'Gamma', 'tan', 'sin', 'cos', 'cot',
    'sec', 'csc', 'ln', 'log', 'times', 'div', 'cdot', 'circ', 'infty',
    'leq', 'geq', 'neq', 'approx', 'rightarrow', 'leftarrow', 'Rightarrow',
    'sum', 'int', 'partial', 'nabla', 'vec', 'hat', 'bar', 'overline',
    'underline', 'text', 'left', 'right', 'mu', 'pi', 'phi'
  ].sort((a, b) => b.length - a.length); // longest-first so "dfrac" wins over "frac"
  const cmdPattern = knownLatexCommands.join('|');

  let out = rawText;

  // 1) Double any single backslash immediately followed by a known LaTeX
  //    command name (skip ones already doubled, via negative lookbehind).
  const latexRegex = new RegExp(`(?<!\\\\)\\\\(${cmdPattern})`, 'g');
  out = out.replace(latexRegex, '\\\\$1');

  // 2) A stray \' is never valid JSON — it almost always means Gemini
  //    over-escaped a plain apostrophe (e.g. in "Earth's"). Drop the
  //    backslash entirely rather than doubling it, restoring the intended
  //    plain apostrophe character.
  out = out.replace(/(?<!\\)\\'/g, "'");

  return out;
}

function applyLatexSafetyNet(question) {
  if (!question) return question;
  const patched = { ...question };
  if (typeof patched.question === 'string') {
    patched.question = autoWrapStrayLatex(patched.question);
  }
  if (Array.isArray(patched.options)) {
    patched.options = patched.options.map(opt => typeof opt === 'string' ? autoWrapStrayLatex(opt) : opt);
  }
  if (typeof patched.explanation === 'string') {
    patched.explanation = autoWrapStrayLatex(patched.explanation);
  }
  return patched;
}

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

  // 🚨 buildExclusionBlock is now a function, not a one-time string — it gets
  // rebuilt on every chunk iteration so that questions generated in EARLIER
  // chunks of THIS SAME request are also fed back in as exclusions. Previously
  // this block was built once before the while-loop from only the pre-existing
  // pool rows, so chunk 2 (Qs 16-30) had zero awareness of chunk 1 (Qs 1-15)
  // and could — and did — regenerate the exact same questions.
  function buildExclusionBlock(entries) {
    if (!entries || entries.length === 0) return "";
    // Keep the most recent 50 to bound prompt size even as this list grows
    // across chunk iterations within a single large request.
    const trimmed = entries.slice(-50);
    const exclusionLines = trimmed.map((q, i) => {
      const optsText = Array.isArray(q.options) ? ` Options: ${q.options.join(' | ')}` : '';
      return `${i + 1}. ${q.question_text}${optsText}`;
    }).join('\n');
    return `\n\n[DO NOT REPEAT — EXISTING QUESTIONS IN POOL AND ALREADY GENERATED IN THIS SESSION]\nThese questions already exist for this exact exam/subject/topic/difficulty combination (including ones generated moments ago in an earlier chunk of this very same request). Do NOT generate anything testing the same underlying concept, even if reworded differently:\n${exclusionLines}\n`;
  }

  // Running list that starts with pre-existing pool candidates and grows with
  // every freshly generated + accepted question across chunk iterations.
  let runningExclusionEntries = [...(exclusionCandidates || [])];

  const MAX_CHUNK_SIZE = 15;
  let allCompiled = [];
  // 🚨 Loop now continues until we actually HAVE `count` accepted questions,
  // not just until we've made ceil(count/15) request-chunks. Since duplicates
  // get discarded inside the loop (see below), requesting a fixed chunk size
  // per iteration no longer guarantees enough survive — so we track shortfall
  // against allCompiled.length instead of decrementing a fixed `remaining`.
  let attempts = 0;
  // 🚨 Scales with `count` instead of a fixed 6 — a fixed cap meant a single
  // 15-question batch had only 1 "ideal" attempt of slack, so a high Gemini
  // duplicate-discard rate could exhaust MAX_ATTEMPTS before reaching count.
  // Formula gives at least double the "ideal" number of chunks as headroom,
  // while still bailing out eventually so a stubborn topic can't loop forever.
  const MAX_ATTEMPTS = Math.max(6, Math.ceil(count / MAX_CHUNK_SIZE) * 2);

  while (allCompiled.length < count && attempts < MAX_ATTEMPTS) {
    attempts++;
    const stillNeeded = count - allCompiled.length;
    const chunkSize = Math.min(MAX_CHUNK_SIZE, stillNeeded);
    const sessionSeed = Math.random().toString(36).substring(7);
    const exclusionBlock = buildExclusionBlock(runningExclusionEntries);

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

[LATEX FORMATTING — MANDATORY, NOT OPTIONAL, WHENEVER MATH NOTATION APPEARS]: Any time your question or options text contains a mathematical symbol, you MUST wrap that entire expression in single dollar signs ($...$) using proper LaTeX commands — never output raw ASCII math shorthand. Concretely:
- Powers/exponents: write $x^2$, $\\tan^2\\theta$ — NEVER "x^2" or "tan^2θ" as bare text outside $...$.
- Roots: write $\\sqrt{x}$, $\\sqrt{y^2+2xy}$ — NEVER "√(y^2+2xy)" as bare text.
- Fractions: write $\\frac{a}{b}$ — NEVER "a/b" when it represents a real fraction/ratio (plain sentence fractions like "half of the class" are fine as text).
- Greek letters/trig: write $\\theta$, $\\lambda$, $\\sin\\theta$, $\\cos\\theta$, $\\cot\\theta$ — never the bare Unicode glyphs.
- Subscripts: write $x_1$, $a_n$ — never "x_1" as plain text.
This applies to BOTH the "question" field AND every string inside "options" — if an option is a formula like √(y²+2xy)/(x+y), it must be output as the LaTeX string "$\\sqrt{y^2+2xy}/(x+y)$", not as raw symbols. If a topic is purely conceptual with zero math, do not force LaTeX into it — but the instant ANY exponent, root, fraction-as-ratio, or Greek/trig symbol appears anywhere in the question or options, it is a hard requirement to wrap it, with zero exceptions.

[JSON ESCAPE SAFETY — CRITICAL, APPLIES TO YOUR RAW OUTPUT]: Your entire response is parsed as JSON. Any backslash you write inside a JSON string value MUST be a DOUBLE backslash (\\\\), never a single one — this applies to every LaTeX command (write \\\\frac not \\frac, \\\\sqrt not \\sqrt, \\\\theta not \\theta) and to any other backslash-like character. A single backslash before certain letters (like f, n, t, b, r) breaks JSON parsing or silently corrupts your own output. Also, for non-LaTeX content (e.g. geography coordinates, degree-minute-second notation like 23°26'22", or apostrophes in words like "Earth's"), never insert a backslash before the apostrophe or quote character — write it as a plain straight character; do NOT write \\' or \\".

[SELF-VERIFICATION — MANDATORY BEFORE FINALIZING EACH QUESTION]: Before finalizing each question, work through this internally: (1) SOLVE: Actually derive the correct answer yourself from first principles for this specific question — compute it, don't assume it (for patterns/ciphers/sequences: apply the rule to EVERY single element/letter and confirm it holds for ALL of them, not just some; for math: do the actual calculation; for facts: confirm accuracy). (2) MATCH: Confirm your derived answer exactly equals the correctOptionIndex option's text/value — not approximately, exactly. (3) EXPLANATION CHECK: Confirm the explanation you are about to write, if followed literally step by step, actually produces YOUR derived answer and no other option. (4) DISTRACTORS: Confirm the 3 wrong options are plausible (reflect common mistakes) but are clearly wrong once your verified derivation is applied. (5) LATEX CHECK: Scan your own question and options text for any bare ^, √, Greek letter, or a/b-style fraction that is NOT wrapped in $...$ — if found, fix it before output. (6) If any check in (1)-(5) fails, discard this draft internally and construct a cleaner question from scratch — do not output anything that fails this verification. Only the final, verified question objects should appear in your output.

JSON schema: {"questions": [{"question":"","options":["","","",""],"correctOptionIndex":0,"explanation":""}]}.
Explanation: Max 20 words core fact wrapped in LaTeX where needed.`;
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
        const rawJsonSlice = responseText.substring(startBrace, endBrace + 1);
        parsedData = JSON.parse(sanitizeJsonEscapes(rawJsonSlice));
        break;
      } catch (err) {
        retries--;
        if (retries === 0) throw err;
        await new Promise(r => setTimeout(r, 2000));
      }
    }

    if (parsedData?.questions?.length) {
      // 🚨 Per-question duplicate check happens HERE now, immediately after
      // each chunk, against the running exclusion list (pool history +
      // everything accepted so far in this request). This is what actually
      // stops chunk 2 from repeating chunk 1 — filtering only at the very
      // end (as build-test's outer isLikelyDuplicate pass does) was too late,
      // since by then both chunks were already merged and it only compared
      // against pool history, not against each other.
      for (const rawQ of parsedData.questions) {
        const asPoolShape = { question_text: rawQ.question, options: rawQ.options };
        if (isLikelyDuplicate(asPoolShape, runningExclusionEntries)) {
          console.warn("⚠️ Discarded an intra-request duplicate (matched an earlier chunk or pool history).");
          continue;
        }
        const q = applyLatexSafetyNet(rawQ);
        allCompiled.push(q);
        runningExclusionEntries.push({ question_text: q.question, options: q.options });
      }
    }
  }

  return allCompiled.slice(0, count);
}

// ======================================================================
// 🎯 ROUTE 3 (NEW): SERVE QUESTIONS FROM POOL (with AI fallback)
// Pool-first, AI-fallback-with-exclusion-list design.
// Answers are ALWAYS stripped before the response leaves this route —
// this is where "Layer 2" of the answer-hiding security actually lives.
// ======================================================================
app.post('/api/pool/serve-questions', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified, never trust req.body.studentId again
    const { exam, subject, topic, difficulty, type, count, language, origin } = req.body;

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
        .eq('type', qType)
        .order('created_at', { ascending: false })
        .limit(50);
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
app.post('/api/pool/submit-attempt', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified
    const { questionId, selectedOptionIndex } = req.body;

    if (!questionId) {
      return res.status(400).json({ success: false, error: "questionId missing." });
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
app.post('/api/pool/toggle-save', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified
    const { questionId, saved } = req.body;

    if (!questionId || typeof saved !== 'boolean') {
      return res.status(400).json({ success: false, error: "questionId and saved (boolean) are required." });
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
// 🎯 ROUTE (NEW): DEDUCT A BRAINFEED CREDIT (called at load time, not at
// completion). Called once when a fresh 15-question batch loads, and
// again if the student uses "Load More" (max 2 calls per user-facing
// session, since Load More is capped at one use). Deducting at load time
// — rather than at completion — means a credit is spent the moment the
// questions are served, so a student can't dodge the charge by closing
// the tab before finishing. Each call is its own independent ledger row,
// even though multiple calls can share the same sessionUUID as their
// reference (both point to the same logical session).
// ======================================================================
// Shared helper: builds the same human-readable session label used both when
// a credit is deducted (credit_transactions.session_label) and when the
// session's own data is saved (brainfeed_sessions.session_label) — kept as
// one function so the two labels for the same session never drift apart.
function buildBrainfeedSessionLabel(exam, subjectSection, subject) {
  return [exam, subjectSection, subject].filter(Boolean).length > 0
    ? `${exam || 'BrainFeed'} — ${subjectSection || ''}${subject ? ': ' + subject : ''}`.trim()
    : 'BrainFeed Session';
}

app.post('/api/brainfeed/deduct-credit', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified
    const { sessionUUID, exam, subjectSection, subject } = req.body;

    if (!sessionUUID) {
      return res.status(400).json({ success: false, error: "sessionUUID is required." });
    }

    const sessionLabel = buildBrainfeedSessionLabel(exam, subjectSection, subject);

    const { data: profile, error: profileReadErr } = await supabase
      .from('profiles')
      .select('brainfeed_credits')
      .eq('id', studentId)
      .single();

    if (profileReadErr) throw profileReadErr;

    const oldCredits = profile?.brainfeed_credits || 0;
    const newCredits = oldCredits - 1; // tracked only — not enforced/blocked yet, can go negative for now

    const { error: ledgerErr } = await supabase
      .from('credit_transactions')
      .insert({
        user_id: studentId,
        feature: 'brainfeed',
        type: 'consumption',
        amount: -1,
        balance_after: newCredits,
        reference: sessionUUID,
        session_label: sessionLabel
      });

    if (ledgerErr) throw ledgerErr;

    const { error: profileUpdateErr } = await supabase
      .from('profiles')
      .update({ brainfeed_credits: newCredits })
      .eq('id', studentId);

    if (profileUpdateErr) throw profileUpdateErr;

    res.json({ success: true, updatedBrainfeedCredits: newCredits });

  } catch (error) {
    console.error("❌ Deduct BrainFeed Credit Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to deduct credit." });
  }
});

// ======================================================================
// 🎯 ROUTE (NEW): SAVE / UPDATE A BRAINFEED SESSION'S DATA
// No credit logic here anymore (moved to /deduct-credit above, which runs
// at load time). This route just persists the session content:
//   - attempts_ledger upserts (submit-attempt) are completely untouched
//     and keep happening exactly as before, per-question, unrelated to this.
//   - question_ids + answers only — correct answers/explanations are
//     re-fetched from question_pool at revise-time (AnalysisPortal pattern).
//   - Updates the existing cumulative brainfeed_count/brainfeed_accuracy
//     columns on profiles (same aggregation math as before).
// INSERT-OR-UPDATE: the row's id is the client-generated sessionUUID
// (the same one used for the credit deduction reference above), not an
// auto-generated id. This lets "Save and Exit" create a partial row, and
// later completing that same session (after "Continue Session") update
// that same row instead of creating a duplicate.
// ======================================================================
app.post('/api/brainfeed/complete-session', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified
    const { sessionUUID, questionIds, answers, attempted, correct, isCompleted, exam, subjectSection, subject } = req.body;

    if (!sessionUUID) {
      return res.status(400).json({ success: false, error: "sessionUUID is required." });
    }
    if (!Array.isArray(questionIds) || !Array.isArray(answers)) {
      return res.status(400).json({ success: false, error: "questionIds and answers arrays are required." });
    }
    if (typeof attempted !== 'number' || typeof correct !== 'number') {
      return res.status(400).json({ success: false, error: "attempted and correct counts are required." });
    }
    if (attempted === 0) {
      return res.status(400).json({ success: false, error: "Nothing was attempted in this session." });
    }

    const sessionAccuracy = Math.round((correct / attempted) * 100);
    const sessionLabel = buildBrainfeedSessionLabel(exam, subjectSection, subject);

    // 🐛 FIX: if this session was already partially saved once (e.g. "Save and
    // Exit"), profiles.brainfeed_count/accuracy already counted that partial
    // attempt. Completing the same session later must apply only the DELTA
    // (attempted - previouslyAttempted), not the full new total again, or
    // those earlier questions get double-counted into the cumulative stats.
    const { data: existingSession } = await supabase
      .from('brainfeed_sessions')
      .select('score, answers')
      .eq('id', sessionUUID)
      .eq('user_id', studentId)
      .maybeSingle();

    const previouslyAttempted = existingSession
      ? (existingSession.answers || []).filter(a => a !== null && a !== undefined).length
      : 0;
    const previouslyCorrect = existingSession ? (existingSession.score || 0) : 0;

    // Insert-or-update keyed on the client-generated sessionUUID.
    // upsert() with an explicit id handles both "first save" (Save and Exit,
    // or a direct finish) and "second save" (finishing after a resume)
    // without us needing to check existence first.
    const { data: sessionRow, error: sessionErr } = await supabase
      .from('brainfeed_sessions')
      .upsert({
        id: sessionUUID,
        user_id: studentId,
        question_ids: questionIds,
        answers: answers,
        score: correct,
        accuracy: sessionAccuracy,
        is_completed: !!isCompleted,
        session_label: sessionLabel
      })
      .select()
      .single();

    if (sessionErr) throw sessionErr;

    // Cumulative stats update — now uses only the DELTA since the last save
    // of this same session, so a partial "Save and Exit" followed later by
    // a full completion doesn't double-count the questions from the first save.
    const deltaAttempted = Math.max(0, attempted - previouslyAttempted);
    const deltaCorrect = Math.max(0, correct - previouslyCorrect);

    const { data: profile, error: profileReadErr } = await supabase
      .from('profiles')
      .select('brainfeed_count, brainfeed_accuracy')
      .eq('id', studentId)
      .single();

    if (profileReadErr) throw profileReadErr;

    const oldAttempted = profile?.brainfeed_count || 0;
    const oldAccuracy = profile?.brainfeed_accuracy || 0;
    const oldCorrect = Math.round((oldAccuracy / 100) * oldAttempted);
    const newTotalQuestions = oldAttempted + deltaAttempted;
    const newTotalCorrect = oldCorrect + deltaCorrect;
    const newOverallAccuracy = newTotalQuestions > 0 ? Math.round((newTotalCorrect / newTotalQuestions) * 100) : 0;

    const { error: profileUpdateErr } = await supabase
      .from('profiles')
      .update({
        brainfeed_count: newTotalQuestions,
        brainfeed_accuracy: newOverallAccuracy
      })
      .eq('id', studentId);

    if (profileUpdateErr) throw profileUpdateErr;

    res.json({
      success: true,
      sessionId: sessionRow.id,
      metricsSummary: {
        sessionAccuracy,
        beforeAccuracy: oldAccuracy,
        newAccuracy: newOverallAccuracy,
        attempted,
        correct
      }
    });

  } catch (error) {
    console.error("❌ Complete BrainFeed Session Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to save session." });
  }
});

// ======================================================================
// 🎯 ROUTE (NEW): BEST-EFFORT SESSION SAVE VIA sendBeacon
// navigator.sendBeacon() cannot set custom headers (no Authorization
// header), so this route accepts the Supabase access token inside the
// JSON body instead and verifies it manually — same underlying check
// requireAuth does (supabase.auth.getUser(token)), just not as middleware.
// This exists purely so a tab close / app backgrounding mid-session can
// fire a save that actually has a chance of completing before the page
// is torn down (sendBeacon is designed to survive that, unlike fetch).
// Saves as incomplete (is_completed: false) — a genuine finish always
// goes through the normal /complete-session call above.
// ======================================================================
app.post('/api/brainfeed/beacon-save', async (req, res) => {
  try {
    const { accessToken, sessionUUID, questionIds, answers, attempted, correct, exam, subjectSection, subject } = req.body || {};

    if (!accessToken) {
      return res.status(401).json({ success: false, error: "Missing access token." });
    }

    const { data: authData, error: authErr } = await supabase.auth.getUser(accessToken);
    if (authErr || !authData?.user) {
      return res.status(401).json({ success: false, error: "Invalid or expired session." });
    }
    const studentId = authData.user.id; // ✅ server-verified, same as requireAuth

    if (!sessionUUID || !Array.isArray(questionIds) || !Array.isArray(answers)) {
      return res.status(400).json({ success: false, error: "sessionUUID, questionIds, and answers are required." });
    }
    if (typeof attempted !== 'number' || attempted === 0) {
      // Nothing was answered yet — nothing meaningful to save. Not an error,
      // just a no-op (e.g. the tab closed before the first answer was picked).
      return res.json({ success: true, skipped: true });
    }

    const safeCorrect = typeof correct === 'number' ? correct : 0;
    const sessionAccuracy = Math.round((safeCorrect / attempted) * 100);
    const sessionLabel = buildBrainfeedSessionLabel(exam, subjectSection, subject);

    const { error: sessionErr } = await supabase
      .from('brainfeed_sessions')
      .upsert({
        id: sessionUUID,
        user_id: studentId,
        question_ids: questionIds,
        answers: answers,
        score: safeCorrect,
        accuracy: sessionAccuracy,
        is_completed: false,
        session_label: sessionLabel
      });

    if (sessionErr) throw sessionErr;

    res.json({ success: true });

  } catch (error) {
    console.error("❌ Beacon Save Error:", error);
    // sendBeacon doesn't read the response anyway, but keep this consistent.
    res.status(500).json({ success: false, error: error.message || "Beacon save failed." });
  }
});

// ======================================================================
// 🎯 ROUTE (NEW): BRAINFEED HISTORY — list past sessions + current credits
// Used by the "Revise Previous Sessions" card. Returns lightweight
// metadata only (no question content) so the list loads fast; full
// question data is fetched separately when the user opens a specific
// session to revise, via /api/pool/questions-by-ids (already exists).
// ======================================================================
app.get('/api/brainfeed/history', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified

    const { data: sessions, error: sessionsErr } = await supabase
      .from('brainfeed_sessions')
      .select('id, question_ids, answers, score, accuracy, created_at, is_completed, session_label')
      .eq('user_id', studentId)
      .order('created_at', { ascending: false })
      .limit(50);

    if (sessionsErr) throw sessionsErr;

    const { data: profile, error: profileErr } = await supabase
      .from('profiles')
      .select('brainfeed_credits')
      .eq('id', studentId)
      .single();

    if (profileErr) throw profileErr;

    const formattedSessions = (sessions || []).map(s => ({
      id: s.id,
      questionCount: Array.isArray(s.question_ids) ? s.question_ids.length : 0,
      questionIdsRaw: s.question_ids || [],
      answersRaw: s.answers || [],
      score: s.score,
      accuracy: s.accuracy,
      createdAt: s.created_at,
      isCompleted: !!s.is_completed,
      sessionLabel: s.session_label || 'BrainFeed Session'
    }));

    res.json({
      success: true,
      sessions: formattedSessions,
      brainfeedCredits: profile?.brainfeed_credits || 0
    });

  } catch (error) {
    console.error("❌ BrainFeed History Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to fetch history." });
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
app.post('/api/pool/build-test', requireAuth, rateLimitBuildTest, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified
    const { exam, subject, topic, difficulty, type, count, language, origin, revealAnswers, skipResurfacing, excludeIds } = req.body;

    if (!subject) return res.status(400).json({ success: false, error: "Subject/Section missing bhai!" });

    const targetExam = normalizeTag(exam) || "COMPETITIVE EXAM";
    const targetSubject = normalizeTag(subject);
    const targetTopic = normalizeTag(topic) || null;
    const diffLevel = difficulty || "Medium";
    const qType = type || "Objective";
    const lang = language || "English";
    const rawRequested = parseInt(count) || 5;
    // 🚨 Cap raised from 50 -> 200 to support full-length mock papers (e.g.
    // SSC CPO/CGL/CHSL run 100-200 questions). AI Labs currently only ever
    // sends up to 15 per call (client-side batching loop), so this mainly
    // future-proofs any caller that requests a larger count in one shot.
    const totalRequested = Math.min(200, Math.max(1, rawRequested));

    // 🚨 excludeIds: question_pool row IDs already served earlier in THIS
    // SAME generation session (e.g. AI Labs' batching loop calls build-test
    // once per 15 questions for a 200-question paper). Without this, every
    // subsequent batch call re-runs the same pool query, finds the previous
    // batch's freshly-inserted rows sitting as "unseen" (no ledger entry yet
    // since the student hasn't attempted them), and re-serves them instead
    // of generating new ones — DB stays stuck at 15 rows, same set repeats.
    const excludeIdSet = new Set(Array.isArray(excludeIds) ? excludeIds : []);

    // STEP 1: Pull candidate pool rows matching this exact tag combination.
    let poolQuery = supabase
      .from('question_pool')
      .select('*')
      .eq('target_exam', targetExam)
      .eq('subject', targetSubject)
      .eq('difficulty', diffLevel)
      .eq('type', qType);

    poolQuery = targetTopic ? poolQuery.eq('topic', targetTopic) : poolQuery.is('topic', null);

    const { data: rawCandidatePool, error: poolErr } = await poolQuery.limit(500);
    if (poolErr) throw poolErr;

    // Filtered in JS (not via a Supabase .not('id','in',...) clause) so an
    // empty/undefined excludeIds array is always a safe no-op.
    const candidatePool = excludeIdSet.size > 0
      ? (rawCandidatePool || []).filter(q => !excludeIdSet.has(q.id))
      : (rawCandidatePool || []);

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

    // 🔁 RESURFACING SKIP: "Load More" inside an already-running BrainFeed
    // session sends skipResurfacing=true. Without this, a question marked
    // wrong seconds ago (in the batch that JUST ended) would immediately
    // resurface in the very next batch of the SAME session — the student
    // sees "new" questions that are really just the previous 15 again.
    // Fresh session starts (BrainFeed reopened, AI Labs build) omit this
    // flag entirely, so wrong-answer resurfacing keeps working as before.
    // NOTE: this only works correctly if the ledger is fully up to date by
    // the time this query runs — see the frontend fix that awaits pending
    // submit-attempt writes before firing the Load More request.
    const incorrectQuestions = skipResurfacing ? [] : candidatePool.filter(q => incorrectIds.has(q.id));
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
        .eq('type', qType)
        .order('created_at', { ascending: false })
        .limit(50);
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

    // Gemini returns 503/429 when the model is overloaded or we're being rate
    // limited on their side. That's temporary and not the student's fault, so
    // send back something they can actually act on instead of the raw SDK error.
    const rawMessage = error?.message || '';
    const isUpstreamBusy = rawMessage.includes('503') || rawMessage.includes('Service Unavailable') || rawMessage.includes('overloaded') || rawMessage.includes('high demand') || rawMessage.includes('429');

    if (isUpstreamBusy) {
      return res.status(503).json({
        success: false,
        upstreamBusy: true,
        error: "Our question generator is under heavy load right now. Please try again in a few minutes."
      });
    }

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
app.get('/api/pool/saved-questions', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified

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
// 🎯 ROUTE (NEW): FETCH question_pool ROWS BY ID — for AnalysisPortal
// AnalysisPortal reconstructs a completed AI Labs attempt's full question
// content (text/options/correct answer/explanation) from the question_ids
// array already stored in test_sessions. It used to do this via a direct
// frontend Supabase client query — but question_pool has RLS enabled with
// ZERO policies defined, so that direct client query always silently
// returned zero rows (this is the same lockdown pattern already applied
// deliberately to mock_tests for regular students). This route does the
// same lookup server-side using the service-role key, which bypasses RLS,
// matching the same cloud-based pattern already used by /api/tests/load.
// Ownership check: only returns pool rows for question_ids the student
// actually owns via a completed test_sessions row (prevents an authenticated
// user from fetching arbitrary pool content by guessing/enumerating ids).
// ======================================================================
app.post('/api/pool/questions-by-ids', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified
    const { attemptId, questionIds } = req.body;

    if (!attemptId || !Array.isArray(questionIds) || questionIds.length === 0) {
      return res.status(400).json({ success: false, error: "attemptId and a non-empty questionIds array are required." });
    }

    // Ownership check — this attempt must belong to the verified student.
    const { data: sessionRow, error: sessionErr } = await supabase
      .from('test_sessions')
      .select('id, user_id, question_ids')
      .eq('id', attemptId)
      .eq('user_id', studentId)
      .single();

    if (sessionErr || !sessionRow) {
      return res.status(403).json({ success: false, error: "This attempt does not belong to you, or could not be found." });
    }

    // Only ever fetch ids that are actually part of this owned attempt —
    // never trust the questionIds array from the client body blindly.
    const ownedIds = new Set((sessionRow.question_ids || []).map(String));
    const safeIds = questionIds.filter(id => ownedIds.has(String(id)));

    if (safeIds.length === 0) {
      return res.json({ success: true, questions: [] });
    }

    const { data: poolRows, error: poolErr } = await supabase
      .from('question_pool')
      .select('*')
      .in('id', safeIds);

    if (poolErr) throw poolErr;

    res.json({ success: true, questions: poolRows || [] });

  } catch (error) {
    console.error("❌ Questions-By-Ids Fetch Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to fetch question content." });
  }
});

// ======================================================================
// 🎯 ROUTE (NEW): BRAINFEED QUESTIONS-BY-IDS (for Revise Previous Sessions)
// Same ownership-checked pattern as /api/pool/questions-by-ids above, but
// checks brainfeed_sessions instead of test_sessions — the two are kept
// as separate routes on purpose so the existing AI Labs route above stays
// completely untouched.
// ======================================================================
app.post('/api/brainfeed/questions-by-ids', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified
    const { sessionId, questionIds } = req.body;

    if (!sessionId || !Array.isArray(questionIds) || questionIds.length === 0) {
      return res.status(400).json({ success: false, error: "sessionId and a non-empty questionIds array are required." });
    }

    // Ownership check — this BrainFeed session must belong to the verified student.
    const { data: sessionRow, error: sessionErr } = await supabase
      .from('brainfeed_sessions')
      .select('id, user_id, question_ids')
      .eq('id', sessionId)
      .eq('user_id', studentId)
      .single();

    if (sessionErr || !sessionRow) {
      return res.status(403).json({ success: false, error: "This session does not belong to you, or could not be found." });
    }

    // Only ever fetch ids that are actually part of this owned session.
    const ownedIds = new Set((sessionRow.question_ids || []).map(String));
    const safeIds = questionIds.filter(id => ownedIds.has(String(id)));

    if (safeIds.length === 0) {
      return res.json({ success: true, questions: [] });
    }

    const { data: poolRows, error: poolErr } = await supabase
      .from('question_pool')
      .select('*')
      .in('id', safeIds);

    if (poolErr) throw poolErr;

    res.json({ success: true, questions: poolRows || [] });

  } catch (error) {
    console.error("❌ BrainFeed Questions-By-Ids Fetch Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to fetch question content." });
  }
});

// ======================================================================
// 🎯 ROUTE (NEW): SIGNED URLS FOR PRIVATE SUBJECTIVE UPLOADS
// The 'subjective-uploads' Storage bucket is private (handwritten answer
// sheets are personal student content). AnalysisPortal only has the
// storage PATH (e.g. "{user_id}/{attemptId}/{q}_{i}_{name}") — it can't
// render that directly, so it asks this route for a short-lived signed
// URL. Ownership check: the path's first segment must equal the verified
// user's own id, since paths are always written as {user_id}/... — this
// stops one student from requesting another student's image path.
// ======================================================================
app.post('/api/storage/signed-url', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified
    const { path } = req.body;

    if (!path || typeof path !== 'string') {
      return res.status(400).json({ success: false, error: "path is required." });
    }

    const ownerSegment = path.split('/')[0];
    if (ownerSegment !== studentId) {
      return res.status(403).json({ success: false, error: "This file does not belong to you." });
    }

    const { data, error } = await supabase
      .storage
      .from('subjective-uploads')
      .createSignedUrl(path, 60 * 10); // 10 minutes — plenty for viewing in AnalysisPortal

    if (error) throw error;

    res.json({ success: true, signedUrl: data.signedUrl });

  } catch (error) {
    console.error("❌ Signed URL Generation Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to generate signed URL." });
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
app.post('/api/pool/grade-test', requireAuth, async (req, res) => {
  try {
    const studentId = req.verifiedUserId; // ✅ server-verified
    const { testId, answers } = req.body;

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
    // mock_tests row's own embedded question content (Custom Builder path).
    // Checks BOTH shapes — flat questions_list (non-sectional) and
    // sections[].questions (sectional tests).
    const unresolvedIds = answers.map(a => String(a.questionId)).filter(id => !poolAnswerMap[id]);
    let mockTestsAnswerMap = {};
    if (unresolvedIds.length > 0 && testId) {
      const { data: mockRow, error: mockErr } = await supabase
        .from('mock_tests')
        .select('questions_list, sections')
        .eq('id', testId)
        .maybeSingle();

      if (!mockErr && mockRow) {
        const registerQuestion = (q) => {
          if (unresolvedIds.includes(String(q.id))) {
            const correctVal = q.correct !== undefined ? q.correct : q.correctOptionIndex;
            mockTestsAnswerMap[String(q.id)] = { correctOptionIndex: correctVal, explanation: q.explanation };
          }
        };
        if (Array.isArray(mockRow.questions_list)) mockRow.questions_list.forEach(registerQuestion);
        if (Array.isArray(mockRow.sections)) {
          mockRow.sections.forEach(sec => {
            if (Array.isArray(sec.questions)) sec.questions.forEach(registerQuestion);
          });
        }
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

// ======================================================================
// 🧠 HELPER: Deep-strip correct answers from a mock_tests row's question
// content, handling BOTH shapes that exist in the table — flat
// questions_list (non-sectional tests) and sections[].questions
// (sectional tests). Used by /api/tests/load when reveal=false.
// ======================================================================
function stripMockTestAnswers(row) {
  const stripped = { ...row };

  if (Array.isArray(stripped.questions_list)) {
    stripped.questions_list = stripped.questions_list.map(q => {
      const { correct, correctOptionIndex, explanation, ...rest } = q;
      return rest;
    });
  }

  if (Array.isArray(stripped.sections)) {
    stripped.sections = stripped.sections.map(sec => {
      if (!Array.isArray(sec.questions)) return sec;
      return {
        ...sec,
        questions: sec.questions.map(q => {
          const { correct, correctOptionIndex, explanation, ...rest } = q;
          return rest;
        })
      };
    });
  }

  return stripped;
}

// ======================================================================
// 🎯 ROUTE 9 (NEW): BROWSE TEST SERIES — "Secure Test Delivery" for
// mock_tests. Returns every row's metadata (category/series/section
// structure, title, time, question count) but NEVER the actual question
// content — browsing the catalogue should never leak an entire table's
// worth of answer keys. Powers TestSeries.jsx's category/series/section
// tree, which only ever needs counts and titles to render.
// ======================================================================
app.get('/api/tests/browse', async (req, res) => {
  try {
    const { data, error } = await supabase
      .from('mock_tests')
      .select('id, category_name, series_name, sub_section, sub_group, title, questions, time, has_sectional_timing, created_at')
      .order('created_at', { ascending: false });

    if (error) throw error;

    res.json({ success: true, tests: data || [] });

  } catch (error) {
    console.error("❌ Tests Browse Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to browse test series." });
  }
});

// ======================================================================
// 🎯 ROUTE 10 (NEW): LOAD A SINGLE TEST — "Secure Test Delivery" for
// mock_tests. reveal=false (default) strips answers, for starting/
// reattempting a test. reveal=true returns full data, for reviewing an
// already-completed attempt (post-hoc reveal is fine — the student
// already finished that attempt).
// ======================================================================
app.get('/api/tests/load', async (req, res) => {
  try {
    const { testId, reveal } = req.query;
    if (!testId) return res.status(400).json({ success: false, error: "testId missing bhai!" });

    const { data, error } = await supabase
      .from('mock_tests')
      .select('*')
      .eq('id', testId)
      .maybeSingle();

    if (error) throw error;
    if (!data) return res.status(404).json({ success: false, error: "Test not found." });

    const responseTest = (reveal === 'true') ? data : stripMockTestAnswers(data);

    res.json({ success: true, test: responseTest });

  } catch (error) {
    console.error("❌ Test Load Error:", error);
    res.status(500).json({ success: false, error: error.message || "Failed to load test." });
  }
});

app.listen(PORT, () => console.log(`🔥 Production Secure Server running active on port: ${PORT}`));