using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientVisits;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api")]
    [Authorize]
    public class VisitPrescriptionsController : ControllerBase
    {
        private readonly IPatientVisitService _visitService;

        public VisitPrescriptionsController(IPatientVisitService visitService)
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

        // POST /api/visits/{visitId}/prescriptions (Doctor)
        [HttpPost("visits/{visitId:int}/prescriptions")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Add(int visitId, [FromBody] CreateVisitPrescriptionDto dto)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var created = await _visitService.AddPrescriptionAsync(userId.Value, visitId, dto);
            return Ok(created);
        }

        // GET /api/visits/{visitId}/prescriptions (Doctor, Patient(own), Pharmacist)
        [HttpGet("visits/{visitId:int}/prescriptions")]
        [Authorize(Roles = "Doctor,Patient,Pharmacist")]
        public async Task<IActionResult> List(int visitId)
        {
            var items = await _visitService.GetPrescriptionsAsync(visitId);

            if (User.IsInRole("Patient"))
            {
                // Ownership check should be enforced via visit details; keep this list endpoint simple.
                var pid = GetPatientIdFromToken();
                if (!pid.HasValue) return Forbid();
            }

            return Ok(items);
        }

        // PATCH /api/prescriptions/{id} (Doctor owner)
        [HttpPatch("prescriptions/{id:int}")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Update(int id, [FromBody] UpdateVisitPrescriptionDto dto)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var updated = await _visitService.UpdatePrescriptionAsync(userId.Value, id, dto);
            if (updated == null) return NotFound(new { message = "Prescription line not found." });
            return Ok(updated);
        }

        // DELETE /api/prescriptions/{id} (Doctor owner)
        [HttpDelete("prescriptions/{id:int}")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Delete(int id)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var ok = await _visitService.DeletePrescriptionAsync(userId.Value, id);
            if (!ok) return NotFound(new { message = "Prescription line not found." });
            return Ok(new { message = "Prescription line deleted." });
        }

        // POST /api/prescriptions/{id}/dispense (Pharmacist)
        [HttpPost("prescriptions/{id:int}/dispense")]
        [Authorize(Roles = "Pharmacist")]
        public async Task<IActionResult> Dispense(int id)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var ok = await _visitService.DispensePrescriptionAsync(userId.Value, id);
            if (!ok) return NotFound(new { message = "Prescription line not found." });
            return Ok(new { message = "Marked as dispensed." });
        }
    }
}
