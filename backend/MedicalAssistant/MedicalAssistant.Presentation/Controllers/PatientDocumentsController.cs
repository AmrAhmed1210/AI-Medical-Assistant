using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.ComponentModel.DataAnnotations;

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

    // ─────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────

    private int GetPatientIdFromClaims()
    {
        var claim = User.FindFirst("PatientId")?.Value
                    ?? User.FindFirst("UserId")?.Value
                    ?? User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;

        return int.TryParse(claim, out var id) ? id : 0;
    }

    private bool IsDoctorOrOwner(int patientId)
    {
        var role = User.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value
                   ?? string.Empty;

        if (role.Equals("Doctor", System.StringComparison.OrdinalIgnoreCase))
            return true;

        return GetPatientIdFromClaims() == patientId;
    }

    // ─────────────────────────────────────────────
    // Endpoints
    // ─────────────────────────────────────────────

    /// <summary>Get all documents for a patient, optionally filtered by type.</summary>
    [HttpGet("patients/{id:int}/documents")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status403Forbidden)]
    public async Task<IActionResult> GetForPatient(int id, [FromQuery] string? type = null)
    {
        if (!IsDoctorOrOwner(id))
            return Forbid();

        var docs = await _patientRecordService.GetPatientDocumentsAsync(id, type);
        return Ok(docs.OrderByDescending(d => d.UploadedAt));
    }

    /// <summary>Create a document record (no file upload).</summary>
    [HttpPost("patients/{id:int}/documents")]
    [Authorize(Roles = "Doctor,Patient")]
    [ProducesResponseType(StatusCodes.Status201Created)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status403Forbidden)]
    public async Task<IActionResult> CreateForPatient(
        int id,
        [FromBody] CreatePatientDocumentDto dto)
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
            DocumentDate = dto.DocumentDate ?? DateTime.UtcNow,
        };

        var created = await _patientRecordService.AddPatientDocumentAsync(id, document);
        return CreatedAtAction(nameof(GetForPatient), new { id }, created);
    }

    /// <summary>Upload a file and create a document record for a patient.</summary>
    [HttpPost("patients/{id:int}/documents/upload")]
    [Authorize(Roles = "Doctor,Patient")]
    [Consumes("multipart/form-data")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status403Forbidden)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public async Task<IActionResult> UploadForPatient(
        int id,
        [FromForm] UploadDocumentRequest request)
    {
        if (!IsDoctorOrOwner(id))
            return Forbid();

        if (request.File is null || request.File.Length == 0)
            return BadRequest(new { message = "No file provided." });

        var uploadResult = await _photoService.UploadDocumentAsync(request.File);
        if (uploadResult is null)
            return StatusCode(500, new { message = "File upload failed." });

        var document = new PatientDocument
        {
            DocumentType = request.DocumentType ?? "scan",
            Title = request.Title ?? request.File.FileName,
            Description = request.Description,
            FileUrl = uploadResult,
            FileType = request.File.ContentType,
            DocumentDate = DateTime.UtcNow,
        };

        var created = await _patientRecordService.AddPatientDocumentAsync(id, document);
        return Ok(created);
    }

    /// <summary>Update an existing document's metadata.</summary>
    [HttpPatch("patient-documents/{documentId:int}")]
    [Authorize(Roles = "Doctor,Patient")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> Update(
        int documentId,
        [FromBody] UpdatePatientDocumentDto dto)
    {
        if (dto is null)
            return BadRequest(new { message = "Invalid payload." });

        var updates = new PatientDocument
        {
            DocumentType = dto.DocumentType,
            Title = dto.Title,
            Description = dto.Description,
            DocumentDate = dto.DocumentDate ?? default,
        };

        var updated = await _patientRecordService.UpdatePatientDocumentAsync(documentId, updates);
        if (updated is null)
            return NotFound(new { message = "Document not found." });

        return Ok(updated);
    }

    /// <summary>Delete a document by ID.</summary>
    [HttpDelete("patient-documents/{documentId:int}")]
    [Authorize(Roles = "Doctor,Patient,Admin")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> Delete(int documentId)
    {
        var deleted = await _patientRecordService.DeletePatientDocumentAsync(documentId);
        if (!deleted)
            return NotFound(new { message = "Document not found." });

        return Ok(new { message = "Document removed." });
    }
}

/// <summary>Request model for file upload endpoint.</summary>
public sealed class UploadDocumentRequest
{
    [Required]
    public IFormFile File { get; set; } = null!;

    [Required]
    public string DocumentType { get; set; } = "scan";

    [Required]
    public string Title { get; set; } = null!;

    public string? Description { get; set; }
}