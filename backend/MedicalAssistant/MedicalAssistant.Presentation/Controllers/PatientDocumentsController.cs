using System.Linq;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.PatientModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api")]
    [Authorize]
    public class PatientDocumentsController : ControllerBase
    {
        private readonly IPatientRecordService _patientRecordService;
        private readonly IPhotoService _photoService;

        public PatientDocumentsController(IPatientRecordService patientRecordService, IPhotoService photoService)
        {
            _patientRecordService = patientRecordService;
            _photoService = photoService;
        }

        private int GetPatientIdFromClaims()
        {
            var claim = User.FindFirst("PatientId")?.Value
                        ?? User.FindFirst("UserId")?.Value
                        ?? User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;
            return int.TryParse(claim, out var id) ? id : 0;
        }

        private bool IsDoctorOrOwner(int patientId)
        {
            var role = User.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value ?? string.Empty;
            if (role.Equals("Doctor", System.StringComparison.OrdinalIgnoreCase)) return true;
            var currentPatientId = GetPatientIdFromClaims();
            return currentPatientId == patientId;
        }

        // GET /api/patients/{id}/documents
        [HttpGet("patients/{id:int}/documents")]
        public async Task<IActionResult> GetForPatient(int id, [FromQuery] string? type = null)
        {
            if (!IsDoctorOrOwner(id)) return Forbid();
            var docs = await _patientRecordService.GetPatientDocumentsAsync(id, type);
            return Ok(docs.OrderByDescending(d => d.UploadedAt));
        }

        // POST /api/patients/{id}/documents
        [HttpPost("patients/{id:int}/documents")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> CreateForPatient(int id, [FromBody] CreatePatientDocumentDto dto)
        {
            if (!IsDoctorOrOwner(id)) return Forbid();
            if (dto == null) return BadRequest(new { message = "Invalid payload." });

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

        // POST /api/patients/{id}/documents/upload
        [HttpPost("patients/{id:int}/documents/upload")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> UploadForPatient(int id, [FromForm] IFormFile file, [FromForm] string documentType, [FromForm] string title, [FromForm] string? description = null)
        {
            if (!IsDoctorOrOwner(id)) return Forbid();
            if (file == null || file.Length == 0) return BadRequest(new { message = "No file provided." });

            var uploadResult = await _photoService.UploadPhotoAsync(file);
            if (uploadResult == null) return StatusCode(500, new { message = "File upload failed." });

            var document = new PatientDocument
            {
                DocumentType = documentType ?? "scan",
                Title = title ?? file.FileName,
                Description = description,
                FileUrl = uploadResult,
                FileType = file.ContentType,
                DocumentDate = DateTime.UtcNow,
            };

            var created = await _patientRecordService.AddPatientDocumentAsync(id, document);
            return Ok(created);
        }

        // PATCH /api/patient-documents/{id}
        [HttpPatch("patient-documents/{documentId:int}")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> Update(int documentId, [FromBody] UpdatePatientDocumentDto dto)
        {
            if (dto == null) return BadRequest(new { message = "Invalid payload." });

            var updates = new PatientDocument
            {
                DocumentType = dto.DocumentType,
                Title = dto.Title,
                Description = dto.Description,
                DocumentDate = dto.DocumentDate ?? default,
            };

            var updated = await _patientRecordService.UpdatePatientDocumentAsync(documentId, updates);
            if (updated == null) return NotFound(new { message = "Document not found." });
            return Ok(updated);
        }

        // DELETE /api/patient-documents/{id}
        [HttpDelete("patient-documents/{documentId:int}")]
        [Authorize(Roles = "Doctor,Patient,Admin")]
        public async Task<IActionResult> Delete(int documentId)
        {
            var deleted = await _patientRecordService.DeletePatientDocumentAsync(documentId);
            if (!deleted) return NotFound(new { message = "Document not found." });
            return Ok(new { message = "Document removed." });
        }
    }
}
