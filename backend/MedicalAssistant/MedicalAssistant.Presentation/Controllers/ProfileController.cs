using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientDTOs;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    /// <summary>
    /// Patient profile controller.
    /// Provides endpoints for patient to view and update their own profile.
    /// </summary>
    [ApiController]
    [Route("profile")]
    public class ProfileController(IPatientService patientService) : ControllerBase
    {

        /// <summary>
        /// Gets the current patient's profile.
        /// </summary>
        /// <param name="patientId">The patient id (should come from auth token in production)</param>
        /// <returns>The patient profile data</returns>
        [HttpGet("me")]
        [ProducesResponseType(typeof(PatientDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> GetMyProfile([FromQuery] int patientId)
        {
            // TODO: In production, get patientId from User.Claims (authenticated user)
            // var patientId = int.Parse(User.FindFirst("PatientId")?.Value ?? "0");

            var patient = await patientService.GetPatientByIdAsync(patientId);

            if (patient is null)
                return NotFound(new { message = "Patient not found." });

            return Ok(patient);
        }

        /// <summary>
        /// Updates the current patient's profile.
        /// </summary>
        /// <param name="updatePatientDto">The updated patient data</param>
        /// <returns>The updated patient profile</returns>
        [HttpPut("me")]
        [ProducesResponseType(typeof(PatientDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> UpdateMyProfile([FromBody] UpdatePatientDto updatePatientDto)
        {
            // TODO: In production, verify patientId from User.Claims matches updatePatientDto.Id

            if (!ModelState.IsValid)
                return BadRequest(ModelState);

            try
            {
                var updatedPatient = await patientService.UpdatePatientAsync(updatePatientDto);

                if (updatedPatient is null)
                    return NotFound(new { message = "Patient not found." });

                return Ok(updatedPatient);
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new { message = ex.Message });
            }
        }
    }
}
