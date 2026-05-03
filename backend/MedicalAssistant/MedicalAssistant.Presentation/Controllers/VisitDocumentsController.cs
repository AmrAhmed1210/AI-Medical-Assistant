using MedicalAssistant.Services_Abstraction.Contracts;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    public class UploadVisitDocumentRequest
    {
        public IFormFile File { get; set; } = null!;
        public string DocumentType { get; set; } = string.Empty;
        public string Title { get; set; } = string.Empty;
        public string? Description { get; set; }
    }

    [ApiController]
    [Route("api")]
    [Authorize]
    public class VisitDocumentsController : ControllerBase
    {
        private readonly IPatientVisitService _visitService;

        public VisitDocumentsController(IPatientVisitService visitService)
        {
            _visitService = visitService;
        }

        private int? GetUserIdFromToken()
        {
            var uid = User.FindFirst("UserId")?.Value
                   ?? User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;
            return int.TryParse(uid, out var id) ? id : null;
        }

        private int? GetPatientIdFromToken()
        {
            var pid = User.FindFirst("PatientId")?.Value;
            return int.TryParse(pid, out var id) ? id : null;
        }

        // POST /api/visits/{visitId}/documents (Doctor, Nurse)
        [HttpPost("visits/{visitId:int}/documents")]
        [Authorize(Roles = "Doctor,Nurse")]
        [RequestSizeLimit(50_000_000)]
        [Consumes("multipart/form-data")]
        public async Task<IActionResult> Upload(int visitId, [FromForm] UploadVisitDocumentRequest request)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });
            if (request?.File == null || request.File.Length == 0) return BadRequest(new { message = "File is required." });

            var created = await _visitService.UploadDocumentAsync(
                userId.Value,
                visitId,
                request.File,
                request.DocumentType,
                request.Title,
                request.Description);

            return Ok(created);
        }

        // GET /api/visits/{visitId}/documents (Doctor, Patient(own))
        [HttpGet("visits/{visitId:int}/documents")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> List(int visitId)
        {
            var items = await _visitService.GetDocumentsAsync(visitId);

            if (User.IsInRole("Patient"))
            {
                var pid = GetPatientIdFromToken();
                if (!pid.HasValue) return Forbid();
            }

            return Ok(items);
        }

        // DELETE /api/documents/{id} (Doctor owner, Admin)
        [HttpDelete("documents/{id:int}")]
        [Authorize(Roles = "Doctor,Admin")]
        public async Task<IActionResult> Delete(int id)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var ok = await _visitService.DeleteDocumentAsync(userId.Value, id);
            if (!ok) return NotFound(new { message = "Document not found." });
            return Ok(new { message = "Document deleted." });
        }
    }
}
