namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class DoctorUpdateDto
{
    public int SpecialtyId { get; set; }

    public string? Bio { get; set; }

    public decimal? ConsultFee { get; set; }

    public int? YearsExperience { get; set; }

    public string? Location { get; set; }
}
