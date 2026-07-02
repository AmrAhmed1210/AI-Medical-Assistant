using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Shared.DTOs.AIChatDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System.Text.Json.Serialization;
using MedicalAssistant.Services_Abstraction.Contracts;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/[controller]")]
[Authorize]
[Produces("application/json")]
public sealed class ChatController : ControllerBase
{
    private readonly IMedicalAiService _medical;
    private readonly IPatientRecordService _recordService;
    private readonly ILogger<ChatController> _log;

    public ChatController(IMedicalAiService medical, IPatientRecordService recordService, ILogger<ChatController> log)
    {
        _medical = medical;
        _recordService = recordService;
        _log = log;
    }

    private int GetPatientIdFromClaims()
    {
        var claim = User.FindFirst("PatientId")?.Value;
        return int.TryParse(claim, out var id) ? id : 0;
    }

    // ─────────────────────────────────────────────────────────────
    // POST api/chat/ask
    // Simple chat — returns a plain reply string
    // ─────────────────────────────────────────────────────────────
    [HttpPost("ask")]
    [ProducesResponseType(typeof(ChatResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> Ask(
        [FromBody] ChatRequest req,
        CancellationToken ct)
    {
        if (!IsValidRequest(req, out var validationError))
            return BadRequest(new ErrorResponse(validationError));

        _log.LogInformation(
            "[ChatController] ask — question='{Question}'",
            Truncate(req.Question, 80));

        var result = await _medical.AskDetailedAsync(req.Question.Trim(), req.History, ct);

        if (result is null)
        {
            _log.LogWarning("[ChatController] ask — AI service returned null.");
            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("الخدمة غير متاحة مؤقتاً. يرجى المحاولة لاحقاً."));
        }

        return Ok(new ChatResponse(result.Reply ?? string.Empty));
    }

    // ─────────────────────────────────────────────────────────────
    // POST api/chat/ask-detailed
    // Full RAG response — returns the complete AIResponseDTO
    // ─────────────────────────────────────────────────────────────
    [HttpPost("ask-detailed")]
    [ProducesResponseType(typeof(AIResponseDTO), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> AskDetailed(
        [FromBody] ChatRequest req,
        CancellationToken ct)
    {
        if (!IsValidRequest(req, out var validationError))
            return BadRequest(new ErrorResponse(validationError));

        _log.LogInformation(
            "[ChatController] ask-detailed — question='{Question}'",
            Truncate(req.Question, 80));

        var result = await _medical.AskDetailedAsync(req.Question.Trim(), req.History, ct);

        if (result is null)
        {
            _log.LogWarning("[ChatController] ask-detailed — AI service returned null.");
            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("الخدمة غير متاحة مؤقتاً. يرجى المحاولة لاحقاً."));
        }

        return Ok(result);
    }

    // ─────────────────────────────────────────────────────────────
    // POST api/chat/analyze-image
    // Medical image analysis — multipart/form-data
    // ─────────────────────────────────────────────────────────────
    [HttpPost("analyze-image")]
    [ProducesResponseType(typeof(MedicalAnalysisResponseDTO), StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> AnalyzeImage(
        IFormFile? file,
        CancellationToken ct)
    {
        if (file is null || file.Length == 0)
        {
            _log.LogWarning("[ChatController] analyze-image — no file provided.");
            return BadRequest(new ErrorResponse("يرجى إرفاق ملف صورة صحيح."));
        }

        _log.LogInformation(
            "[ChatController] analyze-image — file='{FileName}' size={Size}KB type={Type}",
            file.FileName,
            file.Length / 1024,
            file.ContentType);

        string? patientContext = null;
        var patientId = GetPatientIdFromClaims();
        if (patientId > 0)
        {
            try {
                var medicalHistory = await _recordService.GetPatientCompleteHistoryAsync(patientId);
                patientContext = $"Patient Medical Background:\n" +
                                 $"Chronic Diseases: {string.Join(", ", medicalHistory.ChronicDiseases.Select(d => d.DiseaseName))}\n" +
                                 $"Allergies: {string.Join(", ", medicalHistory.Allergies.Select(a => a.AllergenName))}\n" +
                                 $"Current Medications: {string.Join(", ", medicalHistory.Medications.Select(m => m.MedicationName))}\n";
            } catch { /* proceed without context if fetch fails */ }
        }

        var result = await _medical.AnalyzeMedicalImageAsync(file, patientContext, ct);

        if (result is null)
        {
            _log.LogWarning("[ChatController] analyze-image — AI service returned null.");
            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("تعذّر تحليل الصورة مؤقتاً. يرجى المحاولة لاحقاً."));
        }

        return Ok(result);
    }

    [HttpPost("analyze-history")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> AnalyzeHistory(
        [FromBody] object body,
        CancellationToken ct)
    {
        var result = await _medical.AnalyzePatientHistoryAsync(body, ct);
        if (result is null)
            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("تعذّر تحليل السجل الطبي مؤقتاً."));
        return Ok(result);
    }

    [HttpPost("summarize-surgery")]
    public async Task<IActionResult> SummarizeSurgery(
        [FromBody] SurgerySummaryRequest req,
        CancellationToken ct)
    {
        var result = await _medical.SummarizeSurgeryAsync(req.Description ?? string.Empty, ct);
        return result is null
            ? StatusCode(StatusCodes.Status503ServiceUnavailable, new ErrorResponse("AI service unavailable."))
            : Ok(result);
    }

    [HttpPost("summarize-medical-item")]
    public async Task<IActionResult> SummarizeMedicalItem(
        [FromBody] MedicalItemSummaryRequest req,
        CancellationToken ct)
    {
        var result = await _medical.SummarizeMedicalItemAsync(req.Type ?? "", req.Description ?? "", ct);
        return result is null
            ? StatusCode(StatusCodes.Status503ServiceUnavailable, new ErrorResponse("AI service unavailable."))
            : Ok(result);
    }

    [HttpPost("analyze-vitals")]
    public async Task<IActionResult> AnalyzeVitals([FromBody] object body, CancellationToken ct)
    {
        var result = await _medical.AnalyzeVitalsAsync(body, ct);
        return result is null
            ? StatusCode(StatusCodes.Status503ServiceUnavailable, new ErrorResponse("AI service unavailable."))
            : Ok(result);
    }

    [HttpPost("check-medication-safety")]
    public async Task<IActionResult> CheckMedicationSafety([FromBody] object body, CancellationToken ct)
    {
        var result = await _medical.CheckMedicationSafetyAsync(body, ct);
        return result is null
            ? StatusCode(StatusCodes.Status503ServiceUnavailable, new ErrorResponse("AI service unavailable."))
            : Ok(result);
    }

    [HttpPost("pre-visit-summary")]
    public async Task<IActionResult> GeneratePreVisitSummary([FromBody] PreVisitRequestDto body, CancellationToken ct)
    {
        var result = await _medical.GeneratePreVisitSummaryAsync(body, ct);
        return result is null
            ? StatusCode(StatusCodes.Status503ServiceUnavailable, new ErrorResponse("AI service unavailable."))
            : Ok(result);
    }

    [HttpGet("daily-tip")]
    public async Task<IActionResult> GetDailyTip(CancellationToken ct)
    {
        // STRICT ANONYMIZATION: Patient ID is extracted directly from the secure JWT token claims.
        // It cannot be forged or injected via request body or query parameters.
        var patientId = GetPatientIdFromClaims();
        if (patientId == 0)
        {
            _log.LogWarning("[ChatController] GetDailyTip — Missing or invalid PatientId in claims.");
            return Unauthorized(new ErrorResponse("Unauthorized access."));
        }

        _log.LogInformation("[ChatController] GetDailyTip — Extracting diseases for PatientId: {PatientId}", patientId);

        var chronicDiseases = new List<string>();
        try
        {
            // Only chronic diseases are fetched and passed to the AI to preserve maximum privacy.
            var history = await _recordService.GetPatientCompleteHistoryAsync(patientId);
            chronicDiseases = history.ChronicDiseases.Select(d => d.DiseaseName).ToList();
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "[ChatController] GetDailyTip — Failed to fetch patient history for PatientId: {PatientId}", patientId);
        }

        var payload = new {
            patient_id = patientId.ToString(),
            chronic_diseases = chronicDiseases
        };

        var result = await _medical.GetPersonalizedTipAsync(payload, ct);
        return result is null
            ? StatusCode(StatusCodes.Status503ServiceUnavailable, new ErrorResponse("AI service unavailable."))
            : Ok(result);
    }

    // ─────────────────────────────────────────────────────────────
    // POST api/chat/parse-medical-profile
    // AI-powered medical profile data extraction + auto-save
    // ─────────────────────────────────────────────────────────────
    [HttpPost("parse-medical-profile")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status400BadRequest)]
    [ProducesResponseType(typeof(ErrorResponse), StatusCodes.Status503ServiceUnavailable)]
    public async Task<IActionResult> ParseMedicalProfile(
        [FromBody] ParseMedicalProfileRequest req,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.Text))
            return BadRequest(new ErrorResponse("النص مطلوب."));

        _log.LogInformation("[ChatController] parse-medical-profile — text='{Text}'", req.Text.Length > 80 ? req.Text[..80] + "…" : req.Text);

        // 1. Ask AI to parse the text
        var result = await _medical.ParseMedicalProfileAsync(req.Text.Trim(), ct);
        if (result is null)
            return StatusCode(StatusCodes.Status503ServiceUnavailable,
                new ErrorResponse("تعذّر تحليل البيانات الطبية مؤقتاً."));

        // 2. If saveToProfile is true, auto-save extracted data
        if (req.SaveToProfile)
        {
            var patientId = GetPatientIdFromClaims();
            if (patientId > 0)
            {
                try
                {
                    var json = System.Text.Json.JsonSerializer.Serialize(result);
                    var doc = System.Text.Json.JsonDocument.Parse(json);
                    var root = doc.RootElement;
                    int savedDiseases = 0, savedMeds = 0, savedAllergies = 0;

                    // Save chronic diseases
                    if (root.TryGetProperty("chronic_diseases", out var diseases) && diseases.ValueKind == System.Text.Json.JsonValueKind.Array)
                    {
                        foreach (var d in diseases.EnumerateArray())
                        {
                            try
                            {
                                DateOnly? diagnosedDate = null;
                                if (d.TryGetProperty("diagnosedDate", out var dd) && dd.ValueKind == System.Text.Json.JsonValueKind.String)
                                {
                                    var ddStr = dd.GetString();
                                    if (!string.IsNullOrEmpty(ddStr) && DateOnly.TryParse(ddStr, out var parsed))
                                        diagnosedDate = parsed;
                                }

                                var nameEn = d.TryGetProperty("diseaseName", out var dn) ? dn.GetString() ?? "" : "";
                                var nameAr = d.TryGetProperty("diseaseNameAr", out var dnar) ? dnar.GetString() ?? "" : "";
                                var finalName = string.IsNullOrWhiteSpace(nameAr) ? nameEn : (string.IsNullOrWhiteSpace(nameEn) ? nameAr : $"{nameAr} ({nameEn})");

                                var entity = new MedicalAssistant.Domain.Entities.PatientModule.ChronicDiseaseMonitor
                                {
                                    DiseaseName = finalName,
                                    DiseaseType = d.TryGetProperty("diseaseType", out var dt) ? dt.GetString() ?? "" : "",
                                    Severity = d.TryGetProperty("severity", out var sv) ? sv.GetString() ?? "Moderate" : "Moderate",
                                    DiagnosedDate = diagnosedDate,
                                    IsActive = true,
                                    DoctorNotes = d.TryGetProperty("notes", out var nt) ? nt.GetString() : null,
                                    MonitoringFrequency = "Monthly"
                                };
                                await _recordService.AddChronicDiseaseAsync(patientId, entity);
                                savedDiseases++;
                            }
                            catch (Exception ex) { _log.LogWarning(ex, "Failed to save chronic disease"); }
                        }
                    }

                    // Save medications
                    if (root.TryGetProperty("medications", out var meds) && meds.ValueKind == System.Text.Json.JsonValueKind.Array)
                    {
                        foreach (var m in meds.EnumerateArray())
                        {
                            try
                            {
                                var nameEn = m.TryGetProperty("medicationName", out var mn) ? mn.GetString() ?? "" : "";
                                var nameAr = m.TryGetProperty("medicationNameAr", out var mnar) ? mnar.GetString() ?? "" : "";
                                var finalName = string.IsNullOrWhiteSpace(nameAr) ? nameEn : (string.IsNullOrWhiteSpace(nameEn) ? nameAr : $"{nameAr} ({nameEn})");

                                var entity = new MedicalAssistant.Domain.Entities.PatientModule.MedicationTracker
                                {
                                    MedicationName = finalName,
                                    GenericName = m.TryGetProperty("genericName", out var gn) ? gn.GetString() : null,
                                    Dosage = m.TryGetProperty("dosage", out var ds) ? ds.GetString() ?? "" : "",
                                    Form = m.TryGetProperty("form", out var fm) ? fm.GetString() ?? "Tablet" : "Tablet",
                                    Frequency = m.TryGetProperty("frequency", out var fq) ? fq.GetString() ?? "Once daily" : "Once daily",
                                    Instructions = m.TryGetProperty("instructions", out var ins) ? ins.GetString() : null,
                                    IsChronic = m.TryGetProperty("isChronic", out var ic) && ic.ValueKind == System.Text.Json.JsonValueKind.True,
                                    IsActive = true,
                                    StartDate = DateOnly.FromDateTime(DateTime.UtcNow),
                                    TimesPerDay = 1,
                                    DoseTimes = "08:00"
                                };
                                await _recordService.AddMedicationAsync(patientId, entity);
                                savedMeds++;
                            }
                            catch (Exception ex) { _log.LogWarning(ex, "Failed to save medication"); }
                        }
                    }

                    // Save allergies
                    if (root.TryGetProperty("allergies", out var allergies) && allergies.ValueKind == System.Text.Json.JsonValueKind.Array)
                    {
                        foreach (var a in allergies.EnumerateArray())
                        {
                            try
                            {
                                var nameEn = a.TryGetProperty("allergenName", out var an) ? an.GetString() ?? "" : "";
                                var nameAr = a.TryGetProperty("allergenNameAr", out var anar) ? anar.GetString() ?? "" : "";
                                var finalName = string.IsNullOrWhiteSpace(nameAr) ? nameEn : (string.IsNullOrWhiteSpace(nameEn) ? nameAr : $"{nameAr} ({nameEn})");

                                var entity = new MedicalAssistant.Domain.Entities.PatientModule.AllergyRecord
                                {
                                    AllergenName = finalName,
                                    AllergyType = a.TryGetProperty("allergyType", out var at) ? at.GetString() ?? "Other" : "Other",
                                    Severity = a.TryGetProperty("severity", out var sv) ? sv.GetString() ?? "Moderate" : "Moderate",
                                    ReactionDescription = a.TryGetProperty("reactionDescription", out var rd) ? rd.GetString() : null,
                                    IsActive = true
                                };
                                await _recordService.AddAllergyAsync(patientId, entity);
                                savedAllergies++;
                            }
                            catch (Exception ex) { _log.LogWarning(ex, "Failed to save allergy"); }
                        }
                    }

                    _log.LogInformation(
                        "[ChatController] parse-medical-profile — saved {Diseases} diseases, {Meds} medications, {Allergies} allergies for patient {PatientId}",
                        savedDiseases, savedMeds, savedAllergies, patientId);
                }
                catch (Exception ex)
                {
                    _log.LogError(ex, "[ChatController] Failed to auto-save parsed medical profile data");
                }
            }
        }

        return Ok(result);
    }

    // ─────────────────────────────────────────────────────────────
    // GET api/chat/health
    // ─────────────────────────────────────────────────────────────
    [HttpGet("health")]
    [AllowAnonymous]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public IActionResult Health() =>
        Ok(new { status = "ok", service = "Medical AI API" });

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────
    private static bool IsValidRequest(ChatRequest? req, out string error)
    {
        if (req is null || string.IsNullOrWhiteSpace(req.Question))
        {
            error = "السؤال مطلوب ولا يمكن أن يكون فارغاً.";
            return false;
        }

        if (req.Question.Length > 10000)
        {
            error = "السؤال طويل جداً. الحد الأقصى 10000 حرف.";
            return false;
        }

        error = string.Empty;
        return true;
    }

    private static string Truncate(string value, int maxLength) =>
        value.Length <= maxLength ? value : string.Concat(value.AsSpan(0, maxLength), "…");
}

// ─────────────────────────────────────────────────────────────────
// DTOs — local to presentation layer
// ─────────────────────────────────────────────────────────────────
public sealed record ChatRequest(
    [property: JsonPropertyName("question")]
    string Question,

    [property: JsonPropertyName("history")]
    List<MessageDto>? History
);

public sealed record ChatResponse(
    [property: JsonPropertyName("reply")]
    string Reply
);

public sealed record ErrorResponse(
    [property: JsonPropertyName("error")]
    string Error
);

public sealed record SurgerySummaryRequest(
    [property: JsonPropertyName("description")] string? Description
);

public sealed record MedicalItemSummaryRequest(
    [property: JsonPropertyName("type")] string? Type,
    [property: JsonPropertyName("description")] string? Description
);

public sealed record ParseMedicalProfileRequest(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("saveToProfile")] bool SaveToProfile = false
);

public sealed record PreVisitRequestDto(
    [property: JsonPropertyName("patient_id")] string PatientId,
    [property: JsonPropertyName("age")] int Age,
    [property: JsonPropertyName("gender")] string Gender,
    [property: JsonPropertyName("chief_complaint")] string ChiefComplaint,
    [property: JsonPropertyName("chronic_diseases")] List<string>? ChronicDiseases = null,
    [property: JsonPropertyName("medications")] List<string>? Medications = null,
    [property: JsonPropertyName("allergies")] List<string>? Allergies = null,
    [property: JsonPropertyName("vitals")] List<string>? Vitals = null
);