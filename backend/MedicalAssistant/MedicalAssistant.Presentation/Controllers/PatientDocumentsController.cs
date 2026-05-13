using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api")]
[Authorize]
[Produces("application/json")]
public class PatientDocumentsController : ControllerBase
{
    private readonly IPatientRecordService _patientRecordService;
    private readonly IPhotoService _photoService;

    public PatientDocumentsController(
        IPatientRecordService patientRecordService,
        IPhotoService photoService)
    {
        _patientRecordService = patientRecordService;
        _photoService = photoService;
    }

    private int GetPatientIdFromClaims()
    {
        var claim = User.FindFirst("PatientId")?.Value
                    ?? User.FindFirst("UserId")?.Value
                    ?? User.FindFirst(ClaimTypes.NameIdentifier)?.Value;

        return int.TryParse(claim, out var id) ? id : 0;
    }

    private bool IsDoctorOrOwner(int patientId)
    {
        var role = User.FindFirst(ClaimTypes.Role)?.Value ?? string.Empty;

        if (role.Equals("Doctor", StringComparison.OrdinalIgnoreCase))
            return true;

        return GetPatientIdFromClaims() == patientId;
    }

    [HttpGet("patients/{id:int}/documents")]
    public async Task<IActionResult> GetForPatient(int id, [FromQuery] string? type = null)
    {
        if (!IsDoctorOrOwner(id))
            return Forbid();

        var docs = await _patientRecordService.GetPatientDocumentsAsync(id, type);
        return Ok(docs.OrderByDescending(d => d.UploadedAt));
    }

    [HttpPost("patients/{id:int}/documents")]
    [Authorize(Roles = "Doctor,Patient")]
    public async Task<IActionResult> CreateForPatient(int id, [FromBody] CreatePatientDocumentDto dto)
    {
        if (!IsDoctorOrOwner(id))
            return Forbid();

        if (dto is null)
            return BadRequest(new { message = "Invalid payload." });

        var document = new PatientDocument
        {
            DocumentType = dto.DocumentType ?? string.Empty,
            Title = dto.Title ?? string.Empty,
            Description = dto.Description,
            DocumentDate = dto.DocumentDate ?? DateTime.UtcNow
        };

        var created = await _patientRecordService.AddPatientDocumentAsync(id, document);
        return CreatedAtAction(nameof(GetForPatient), new { id }, created);
    }

    [HttpPost("patients/{id:int}/documents/upload")]
    [Authorize(Roles = "Doctor,Patient")]
    [Consumes("multipart/form-data")]
    public async Task<IActionResult> UploadForPatient(int id, [FromForm] UploadDocumentRequest request)
    {
        if (!IsDoctorOrOwner(id))
            return Forbid();

        if (request.File is null || request.File.Length == 0)
            return BadRequest(new { message = "No file provided." });

        if (string.IsNullOrWhiteSpace(request.Title))
            request.Title = request.File.FileName;

        var uploadResult = await _photoService.UploadDocumentAsync(request.File);

        if (uploadResult is null)
            return StatusCode(500, new { message = "File upload failed." });

        var document = new PatientDocument
        {
            DocumentType = request.DocumentType ?? "scan",
            Title = request.Title,
            Description = request.Description,
            FileUrl = uploadResult,
            FileType = request.File.ContentType,
            DocumentDate = DateTime.UtcNow
        };

        var created = await _patientRecordService.AddPatientDocumentAsync(id, document);
        return Ok(created);
    }

    [HttpPatch("patient-documents/{documentId:int}")]
    [Authorize(Roles = "Doctor,Patient")]
    public async Task<IActionResult> Update(int documentId, [FromBody] UpdatePatientDocumentDto dto)
    {
        if (dto is null)
            return BadRequest(new { message = "Invalid payload." });

        var updates = new PatientDocument
        {
            DocumentType = dto.DocumentType,
            Title = dto.Title,
            Description = dto.Description,
            DocumentDate = dto.DocumentDate ?? default
        };

        var updated = await _patientRecordService.UpdatePatientDocumentAsync(documentId, updates);

        if (updated is null)
            return NotFound(new { message = "Document not found." });

        return Ok(updated);
    }

    [HttpDelete("patient-documents/{documentId:int}")]
    [Authorize(Roles = "Doctor,Patient,Admin")]
    public async Task<IActionResult> Delete(int documentId)
    {
        var deleted = await _patientRecordService.DeletePatientDocumentAsync(documentId);

        if (!deleted)
            return NotFound(new { message = "Document not found." });

        return Ok(new { message = "Document removed." });
    }
}

public sealed class UploadDocumentRequest
{
    public IFormFile? File { get; set; }
    public string DocumentType { get; set; } = "scan";
    public string? Title { get; set; }
    public string? Description { get; set; }
}